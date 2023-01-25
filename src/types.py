from typing import Any, Dict, List, Union

from torch import Tensor

BATCH_OUTPUT = Union[Tensor, Dict[str, Any]]
EPOCH_OUTPUT = List[BATCH_OUTPUT]
