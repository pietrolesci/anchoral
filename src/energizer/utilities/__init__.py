# import inspect
from typing import Any, Dict, List, Union

from lightning_utilities.core.apply_func import apply_to_collection
from numpy import ndarray
from torch import Tensor

# from torch.utils.data import BatchSampler, SequentialSampler
# from src.energizer.enums import RunningStage


def tensor_to_python(t: Tensor, *_) -> Union[ndarray, float, int]:
    """Converts `torch.Tensor` to a `numpy.ndarray` or python scalar type."""
    if t.numel() > 1:
        return t.detach().cpu().numpy()
    return round(t.detach().cpu().item(), 6)


def move_to_cpu(output: Any) -> Any:
    args = (Tensor, tensor_to_python, "cpu")
    return apply_to_collection(output, *args)


def ld_to_dl(ld: List[Dict]) -> Dict[str, List]:
    return {k: [dic[k] for dic in ld] for k in ld[0]}


# def get_hparams() -> Dict:
#     frame = inspect.currentframe().f_back
#     args, _, _, values = inspect.getargvalues(frame)
#     return {arg: values[arg] for arg in args}


# def parse_hparams(hparams: Dict, stage: RunningStage) -> Dict:
#     # filter hparams
#     hparams = {k: v for k, v in hparams.items() if not any([x in k for x in ["self", "loader"]])}

#     # get dataloader hparams
#     loader = hparams.get(f"{stage}_loader", None)
#     if loader is not None:
#         loader = loader.sampler if isinstance(loader.sampler, BatchSampler) else loader
#         loader_hparams = {
#             "batch_size": loader.batch_size,
#             "drop_last": loader.drop_last,
#             "shuffle": not isinstance(loader.sampler, SequentialSampler),
#         }
#         hparams = {**hparams, **loader_hparams}

#     return hparams
