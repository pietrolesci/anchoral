import inspect
from typing import Any, Union

from lightning_utilities.core.apply_func import apply_to_collection
from numpy import ndarray
from torch import Tensor


def tensor_to_python(t: Tensor, *_) -> Union[ndarray, float, int]:
    """Converts `torch.Tensor` to a `numpy.ndarray` or python scalar type."""
    if t.numel() > 1:
        return t.detach().numpy()
    return round(t.item(), 6)


def move_to_cpu(output: Any) -> Any:
    args = (Tensor, tensor_to_python, "cpu")
    return apply_to_collection(output, *args)


def get_hparams():
    frame = inspect.currentframe().f_back
    args, _, _, values = inspect.getargvalues(frame)
    return {arg: values[arg] for arg in args}
