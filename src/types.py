from typing import Dict, Union, Any, Iterable, Mapping

from torch import Tensor
from torchmetrics import Metric


BATCH_OUTPUT = Union[Tensor, Dict]
EVAL_BATCH_OUTPUT = Union[Tensor, Dict, None]
POOL_BATCH_OUTPUT = Dict
METRIC = Union[Metric, Any]
DATASET = Iterable[Mapping]