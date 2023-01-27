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
    epoch: Optional[int] = None
    metrics: Optional[Any] = None
    output: List[BATCH_OUTPUT] = field(default_factory=list)

    def append(self, _x: Union[BATCH_OUTPUT, EVAL_BATCH_OUTPUT]) -> None:
        if _x is None:
            return
        return self.output.append(move_to_cpu(_x))

    def add_metrics(self, _x: Union[MetricCollection, Metric]) -> None:
        self.metrics = move_to_cpu(_x)

    def __repr__(self) -> str:
        self.__class__.__name__
        s = f"{self.__class__.__name__}(epoch={self.epoch}, metrics={self.metrics}, output="
        return f"{s} ..{len(self.output)} batches.. )"

@dataclass
class PoolEpochOutput(EpochOutput):
    topk_scores: ndarray = None
    indices: List[int] = None
    round: int = None

    def from_epoch_output(self, output: EpochOutput) -> None:
        self.output = output.output
        self.metrics = output.metrics


@dataclass
class RunningStageOutput(Output):
    hparams: Optional[Dict] = None

    def __post_init__(self) -> None:
        hparams = {k: v for k, v in self.hparams.items() if k not in ("self",) and not "loader" in k}

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


@dataclass
class FitOutput(RunningStageOutput):
    train: List[EpochOutput] = field(default_factory=list)
    validation: List[EpochOutput] = field(default_factory=list)

    def append(self, _x: EpochOutput, stage: str) -> None:
        return getattr(self, stage).append(_x)


@dataclass
class EvaluationOutput(RunningStageOutput):
    output: EpochOutput = None

    def __post_init__(self) -> None:
        if isinstance(self.output, EpochOutput):
            self.output = self.output.output
        return super().__post_init__()

    def __repr__(self) -> str:
        s = f"{self.__class__.__name__}(hparams={self.hparams}, output="
        if self.output is None:
            return f"{s}None)"
        return f"{s} ..{len(self.output)} batches.. )"


@dataclass
class QueryStrategyOutput(EvaluationOutput):
    scores: List[float] = None
