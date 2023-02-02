from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Union

from torch import Tensor
from torch.utils.data import BatchSampler, DataLoader, SequentialSampler
from torchmetrics import Metric, MetricCollection

from src.enums import RunningStage
from src.types import BATCH_OUTPUT, EVAL_BATCH_OUTPUT
from src.utilities import move_to_cpu
from numpy import ndarray


@dataclass
class Output:
    def to_dict(self) -> Dict:
        return asdict(self)

    def __getitem__(self, key) -> Any:
        return getattr(self, key)


@dataclass
class BatchOutput(Output):
    loss: Tensor
    metrics: Union[MetricCollection, Metric, None] = None


@dataclass
class EpochOutput(Output):
    """Output of a run on an entire dataloader.

    metrics: Metrics aggregated over the entire dataloader.
    output: List of individual outputs at the batch level.
    """
    metrics: Optional[Any] = None
    output: List[BATCH_OUTPUT] = field(default_factory=list)

    def append(self, _x: Union[BATCH_OUTPUT, EVAL_BATCH_OUTPUT, None]) -> None:
        if _x is None:
            return
        return self.output.append(move_to_cpu(_x))

    def add_metrics(self, _x: Union[MetricCollection, Metric]) -> None:
        self.metrics = move_to_cpu(_x)

    def __repr__(self) -> str:
        self.__class__.__name__
        s = f"{self.__class__.__name__}(metrics={self.metrics}, output="
        return f"{s} ..{len(self.output)} batches.. )"


@dataclass
class FitEpochOutput(Output):
    epoch: int
    train: EpochOutput = None
    validation: EpochOutput = None

@dataclass
class RunningStageOutput(Output):
    hparams: Optional[Dict] = None
    output: List[Any] = field(default_factory=list)

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

    def append(self, _x: Union[Any, None]) -> None:
        if _x is None:
            return
        self.output.append(_x)


@dataclass
class FitOutput(RunningStageOutput):
    output: List[FitEpochOutput] = field(default_factory=list)


@dataclass
class EvaluationOutput(RunningStageOutput):
    output: EpochOutput = None

    def __post_init__(self) -> None:
        if isinstance(self.output, EpochOutput):
            self.output = self.output.output
        return super().__post_init__()

"""
Active learning
"""

@dataclass
class PoolEpochOutput(EpochOutput):
    """Output of a run on an entire pool dataloader.

    metrics: Metrics aggregated over the entire pool dataloader.
    output: List of individual outputs at the batch level.
    topk_scores: TopK scores for the pool instances.
    indices: Indices corresponding to the topk instances to query.
    """
    topk_scores: ndarray = None
    indices: List[int] = None


@dataclass
class RoundOutput(Output):
    round: int
    fit: FitOutput = None
    test: EpochOutput = None
    pool: PoolEpochOutput = None


@dataclass
class ActiveFitOutput(RunningStageOutput):
    output = List[RoundOutput]
    
    def __post_init__(self) -> None:
        ignore = ["self", "datamodule"]
        hparams = {k: v for k, v in self.hparams.items() if not any([x in k for x in ignore])}

        datamodule = self.hparams.get("active_datamodule")
        for stage in RunningStage:
            loader = datamodule.get(f"{stage}_loader", None)
            if loader is None:
                continue
            
            loader_hparams = self._dataloader_hparams(loader)
            loader_hparams = {f"{stage}_{k}": v for k, v in loader_hparams.items()}
            hparams = {**hparams, **loader_hparams}

        self.hparams = hparams
