from typing import Any, Dict, Iterable, List, Mapping, Union

from torch import Tensor
from torchmetrics import Metric

from src.energizer.containers import EpochOutput, RoundOutput

BATCH_OUTPUT = Union[Tensor, Dict]
EVAL_BATCH_OUTPUT = Union[Tensor, Dict, None]
POOL_BATCH_OUTPUT = Dict
METRIC = Union[Metric, Any]
DATASET = Iterable[Mapping]
EPOCH_OUTPUT = Union[EpochOutput, Any, List[POOL_BATCH_OUTPUT]]
FORWARD_OUTPUT = Any
ROUND_OUTPUT = Union[RoundOutput, Any]
