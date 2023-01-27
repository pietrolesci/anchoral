from typing import Dict, Union

from torch import Tensor

BATCH_OUTPUT = Union[Tensor, Dict]
EVAL_BATCH_OUTPUT = Union[Tensor, Dict, None]
POOL_BATCH_OUTPUT = Dict
