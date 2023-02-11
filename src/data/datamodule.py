# Here we define DataModule that work with HuggingFace DATASETs.
# We assume that each dataset is already processed and ready for training.
# Think of the DataModule is the last step of the data preparation pipeline.
#
#   download data -> (process data -> prepare data) -> datamodule -> model
#
# That is, the DataModule is only used to feed data to the model during training
# and evaluation.
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple, Union

import hnswlib as hb
import numpy as np
import torch
from lightning.pytorch.core.mixins.hparams_mixin import HyperparametersMixin
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import BatchSampler, DataLoader, RandomSampler, SequentialSampler

from src.enums import RunningStage
from src.types import DATASET


class DataModule(HyperparametersMixin):
    """DataModule that defines dataloading and indexing logic."""

    _hparams_ignore = ["train_dataset", "val_dataset", "test_dataset"]
    _index: hb.Index

    def __init__(
        self,
        train_dataset: DATASET,
        validation_dataset: Optional[DATASET] = None,
        test_dataset: Optional[DATASET] = None,
        batch_size: Optional[int] = 32,
        eval_batch_size: Optional[int] = 32,
        num_workers: Optional[int] = 0,
        pin_memory: Optional[bool] = True,
        drop_last: Optional[bool] = False,
        persistent_workers: Optional[bool] = False,
        shuffle: Optional[bool] = True,
        seed: Optional[int] = 42,
        replacement: bool = False,
    ) -> None:
        super().__init__()
        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset
        self.test_dataset = test_dataset

        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.persistent_workers = persistent_workers
        self.shuffle = shuffle
        self.seed = seed
        self.replacement = replacement

        self.save_hyperparameters(ignore=self._hparams_ignore)
        self.setup()

    def setup(self) -> None:
        pass

    @property
    def index(self) -> Optional[hb.Index]:
        return self._index

    def load_index(self, path: Union[str, Path], embedding_dim: int) -> None:
        p = hb.Index(space="cosine", dim=embedding_dim)
        p.load_index(str(path))
        self._index = p

    def search_index(
        self, query: np.ndarray, query_size: int, query_in_set: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        # retrieve one additional element if the query is in the set we are looking in
        # because the query itself is returned as the most similar element and we need to remove it
        query_size = query_size + 1 if query_in_set else query_size

        indices, distances = self.index.knn_query(query, query_size)

        if query_in_set:
            # remove the first element retrieved if the query is in the set since it's the element itself
            indices, distances = indices[:, 1:], distances[:, 1:]

        return indices, distances

    def train_loader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            sampler=self.get_sampler(self.train_dataset, RunningStage.TRAIN),
            collate_fn=self.get_collate_fn(RunningStage.TRAIN),
        )

    def validation_loader(self) -> Optional[DataLoader]:
        if self.validation_dataset:
            return DataLoader(
                self.validation_dataset,
                sampler=self.get_sampler(self.validation_dataset, RunningStage.VALIDATION),
                collate_fn=self.get_collate_fn(RunningStage.VALIDATION),
            )

    def test_loader(self) -> Optional[DataLoader]:
        if self.test_dataset:
            return DataLoader(
                self.test_dataset,
                sampler=self.get_sampler(self.test_dataset, RunningStage.TEST),
                collate_fn=self.get_collate_fn(RunningStage.TEST),
            )

    def get_collate_fn(self, stage: Optional[str] = None) -> Optional[Callable]:
        return None

    def get_sampler(self, dataset: DATASET, stage: str) -> BatchSampler:
        batch_size = self.batch_size if stage == RunningStage.TRAIN else self.eval_batch_size
        # NOTE: when the batch_size is bigger than the number of available instances in the dataset
        # we get an `IndexError` for Arrow. Here we avoid this
        batch_size = min(batch_size, len(dataset))
        return _get_sampler(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=self.shuffle if stage == RunningStage.TRAIN else False,
            replacement=self.replacement,
            seed=self.seed,
            drop_last=self.drop_last,
        )

    def show_batch(self, stage: RunningStage = RunningStage.TRAIN) -> Any:
        loader = getattr(self, f"{stage}_loader")()
        return next(iter(loader))


"""
Define as globals otherwise pickle complains when running in multi-gpu
"""


def _pad(inputs: List[int], padding_value: float, max_length: int) -> Tensor:
    # truncate -> convert to tensor -> pad
    return pad_sequence(
        [torch.tensor(t[:max_length]) for t in inputs],
        batch_first=True,
        padding_value=padding_value,
    )


def _get_sampler(
    dataset: DATASET,
    batch_size: int,
    shuffle: bool,
    replacement: bool,
    seed: int,
    drop_last: bool,
) -> BatchSampler:
    """Get a sampler optimizer to work with `datasets.Dataset`.

    ref: https://huggingface.co/docs/datasets/use_with_pytorch
    """

    if not shuffle:
        sampler = SequentialSampler(dataset)
    else:
        g = torch.Generator()
        g.manual_seed(seed)
        sampler = RandomSampler(dataset, generator=g, replacement=replacement)

    return BatchSampler(
        sampler,
        batch_size=batch_size,
        drop_last=drop_last,
    )
