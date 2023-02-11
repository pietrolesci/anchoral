from dataclasses import dataclass, field, fields
from typing import Any, Dict, List, Optional, Tuple, Union

from numpy import ndarray
from torch.utils.data import BatchSampler, DataLoader, SequentialSampler
from torchmetrics import Metric, MetricCollection

from src.enums import RunningStage
from src.types import BATCH_OUTPUT, EVAL_BATCH_OUTPUT
from src.utilities import move_to_cpu

# @dataclass
# class Output(dict):
#     def __post_init__(self):
#         for field in fields(self):
#             self[field.name] = getattr(self, field.name)

#     def __getitem__(self, k: Union[str, int]) -> Any:
#         inner_dict = {k: v for (k, v) in self.items()}  # avoid recursion
#         return inner_dict[k]

#     def __setattr__(self, name: str, value: Any) -> None:
#         value = move_to_cpu(value)

#         if name in self.keys() and value is not None:
#             # Don't call self.__setitem__ to avoid recursion errors
#             super().__setitem__(name, value)
#         super().__setattr__(name, value)

#     def to_tuple(self) -> Tuple[Any]:
#         """
#         Convert self to a tuple containing all the attributes/keys that are not `None`.
#         """
#         return tuple(self[k] for k in self.keys())


# @dataclass
# class Output(ModelOutput):
#     time: float = None

# def to_dict(self) -> Dict:
#     return asdict(self)

# def __getitem__(self, key) -> Any:
#     return getattr(self, key)


# @dataclass
# class BatchOutput(Output):
#     batch_idx: int = None
#     output: BATCH_OUTPUT = None


class EpochOutput(list):
    """This is a simple list that moves things to cpu."""

    def append(self, _x: Any) -> None:
        super().append(move_to_cpu(_x))

    def __add__(self, _x: Any) -> None:
        super().__add__(move_to_cpu(_x))

    def __iadd__(self, _x) -> None:
        super().__iadd__(move_to_cpu(_x))


# @dataclass
# class EpochOutput(Output):
#     """Output of a run on an entire dataloader.

#     metrics: Metrics aggregated over the entire dataloader.
#     output: List of individual outputs at the batch level.
#     """

#     metrics: Optional[Any] = None
#     output: List[BatchOutput] = field(default_factory=list)

#     def append(self, _x: Union[BATCH_OUTPUT, EVAL_BATCH_OUTPUT, None]) -> None:
#         if _x is None:
#             return
#         return self.output.append(move_to_cpu(_x))

#     def add_metrics(self, _x: Union[MetricCollection, Metric]) -> None:
#         self.metrics = move_to_cpu(_x)

#     def __repr__(self) -> str:
#         self.__class__.__name__
#         s = f"{self.__class__.__name__}(metrics={self.metrics}, output="
#         return f"{s} ..{len(self.output)} batches.. )"


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
    output: List[FitEpochOutput] = field(default_factory=list)

    def append(self, _x: FitEpochOutput):
        assert isinstance(_x, FitEpochOutput), f"You can only append `FitEpochOutput`s, not {type(_x)}"
        self.output.append(_x)


@dataclass
class EvaluationOutput(MetadataParserMixin):
    output: EpochOutput = None


"""
Active learning
"""


@dataclass
class QueryOutput(EpochOutput):
    """Output of a run on an entire pool dataloader.

    metrics: Metrics aggregated over the entire pool dataloader.
    output: List of individual outputs at the batch level.
    topk_scores: TopK scores for the pool instances.
    indices: Indices corresponding to the topk instances to query.
    """

    topk_scores: ndarray = None
    indices: List[int] = None


@dataclass
class RoundOutput:
    round_idx: int = None
    fit: FitOutput = None
    test: EpochOutput = None
    query: QueryOutput = None


@dataclass
class ActiveFitOutput(MetadataParserMixin):
    output = List[RoundOutput]

    def __post_init__(self) -> None:
        ignore = ["self", "datamodule"]
        hparams = {k: v for k, v in self.hparams.items() if not any([x in k for x in ignore])}

        datamodule = self.hparams.get("active_datamodule")
        self.hparams = {**hparams, **datamodule.hparams}
