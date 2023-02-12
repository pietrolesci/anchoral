from typing import Any, Dict, Iterable, Mapping, Union

from torch import Tensor
from torchmetrics import Metric

from src.containers import EpochOutput

BATCH_OUTPUT = Union[Tensor, Dict]
EVAL_BATCH_OUTPUT = Union[Tensor, Dict, None]
POOL_BATCH_OUTPUT = Dict
METRIC = Union[Metric, Any]
DATASET = Iterable[Mapping]
EPOCH_OUTPUT = Union[EpochOutput, Any]
FORWARD_OUTPUT = Any
