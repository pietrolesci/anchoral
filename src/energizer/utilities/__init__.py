# import inspect
import contextlib
import random
from typing import Any, Dict, Generator, List, Union

import numpy as np
import torch
from lightning.fabric.utilities.seed import _collect_rng_states, _set_rng_states
from lightning_utilities.core.apply_func import apply_to_collection
from numpy import ndarray, generic
from torch import Tensor

# from torch.utils.data import BatchSampler, SequentialSampler
# from src.energizer.enums import RunningStage


def tensor_to_python(t: Tensor, *_) -> Union[ndarray, float, int]:
    """Converts `torch.Tensor` to a `numpy.ndarray` or python scalar type."""
    # if t.numel() > 1:
    return t.detach().cpu().numpy()
    # return round(t.detach().cpu().item(), 6)


def make_dict_json_serializable(d: Dict) -> Dict:
    return {k: round(v.item(), 6) if isinstance(v, (ndarray, generic)) else v for k, v in d.items()}


def move_to_cpu(output: Any) -> Any:
    args = (Tensor, tensor_to_python, "cpu")
    return apply_to_collection(output, *args)


def ld_to_dl(ld: List[Dict]) -> Dict[str, List]:
    return {k: [dic[k] for dic in ld] for k in ld[0]}


@contextlib.contextmanager
def local_seed(seed: int) -> Generator[None, None, None]:
    """A context manager that allows to locally change the seed.

    Upon exit from the context manager it resets the random number generator state
    so that the operations that happen in the context do not affect randomness outside
    of it.
    """
    # collect current states
    states = _collect_rng_states()

    # set seed in the context
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # run code in context
    yield

    # reset states when exiting the context
    _set_rng_states(states)


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
