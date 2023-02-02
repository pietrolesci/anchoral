# Here we define DataModule that work with HuggingFace Datasets.
# We assume that each dataset is already processed and ready for training.
# Think of the DataModule is the last step of the data preparation pipeline.
#
#   download data -> (process data -> prepare data) -> datamodule -> model
#
# That is, the DataModule is only used to feed data to the model during training
# and evaluation.
# In addition, the ActiveDataModule also implements the logic to label data.
import os
import random
from functools import partial
from typing import Callable, Dict, List, Optional, Union

import pandas as pd
import torch
from datasets import Dataset
from torch import Tensor
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerBase

from src.data.datamodule import DataModule, _pad
from src.enums import InputColumns, RunningStage, SpecialColumns


class ActiveDataModule(DataModule):
    _df: pd.DataFrame = None

    """
    Properties
    """

    @property
    def should_val_split(self):
        return self.validation_dataset is None

    @property
    def total_data_size(self) -> int:
        return self.train_size + self.pool_size

    @property
    def train_size(self) -> int:
        return self._df[SpecialColumns.IS_LABELLED].sum()

    @property
    def pool_size(self) -> int:
        return len(self._df) - self.train_size

    @property
    def has_labelled_data(self) -> bool:
        """Checks whether there are labelled data available."""
        return self.train_size > 0

    @property
    def has_unlabelled_data(self) -> bool:
        """Checks whether there are data to be labelled."""
        return self.pool_size > 0

    @property
    def has_test_data(self) -> bool:
        return self.test_dataset is not None and len(self.test_dataset) > 0

    @property
    def last_labelling_round(self) -> int:
        """Returns the number of the last active learning step."""
        return int(self._id[SpecialColumns.LABELLING_ROUND].max())

    @property
    def data_statistics(self) -> Dict[str, int]:
        return {
            "total_data_size": self.total_data_size,
            "train_size": self.train_size,
            "pool_size": self.pool_size,
            "num_train_batches": len(self.train_dataloader()),
            "num_pool_batches": len(self.pool_dataloader()),
        }

    """
    Helper methods
    """

    def setup(self, stage: Optional[str] = None) -> None:
        self._df = (
            self.train_dataset
            # cast to dataframe
            .to_pandas()
            # and create index
            .assign(
                **{
                    SpecialColumns.ID: lambda df_: df_.index.tolist(),
                    SpecialColumns.IS_LABELLED: False,
                    SpecialColumns.IS_VALIDATION: False,
                    SpecialColumns.LABELLING_ROUND: -1,
                }
            )
        )

    """
    Main methods
    """

    def label(self, indices: List[int], round_id: int, val_perc: Optional[float] = None) -> None:
        """Moves instances at index `pool_idx` from the `pool_fold` to the `train_fold`.

        Args:
            pool_idx (List[int]): The index (relative to the pool_fold, not the overall data) to label.
        """
        assert isinstance(indices, list), ValueError(f"`indices` must be of type `List[int]`, not {type(indices)}.")
        assert isinstance(val_perc, float) or val_perc is None, ValueError(
            f"`val_perc` must be of type `float`, not {type(val_perc)}"
        )

        mask = self._df[SpecialColumns.ID].isin(indices)
        self._df.loc[mask, SpecialColumns.IS_LABELLED] = True
        self._df.loc[mask, SpecialColumns.LABELLING_ROUND] = round_id

        if self.should_val_split and val_perc is not None:
            n_val = round(val_perc * len(indices)) or 1
            val_indices = random.sample(indices, n_val)
            self._df.loc[self._df[SpecialColumns.ID].isin(val_indices), SpecialColumns.IS_VALIDATION] = True

    """
    DataLoaders
    """

    def train_loader(self) -> DataLoader:
        train_df = self._df.loc[
            (self._df[SpecialColumns.IS_LABELLED] == True) & (self._df[SpecialColumns.IS_VALIDATION] == False)
        ]
        train_dataset = Dataset.from_pandas(train_df, preserve_index=False)

        return DataLoader(
            train_dataset,
            sampler=self.get_sampler(train_dataset, RunningStage.TRAIN),
            collate_fn=self.get_collate_fn(RunningStage.TRAIN),
        )

    def validation_loader(self) -> Optional[DataLoader]:
        if self.should_val_split:
            val_df = self._df.loc[
                (self._df[SpecialColumns.IS_LABELLED] == True) & (self._df[SpecialColumns.IS_VALIDATION] == True)
            ]
            val_dataset = Dataset.from_pandas(val_df, preserve_index=False)
        else:
            val_dataset = self.validation_dataset

        if val_dataset is not None and len(val_dataset) > 0:
            return DataLoader(
                val_dataset,
                sampler=self.get_sampler(val_dataset, RunningStage.VALIDATION),
                collate_fn=self.get_collate_fn(RunningStage.VALIDATION),
            )

    def pool_loader(self) -> DataLoader:
        pool_df = self._df.loc[
            (self._df[SpecialColumns.IS_LABELLED] == False), [i for i in self._df.columns if i != InputColumns.TARGET]
        ]
        pool_dataset = Dataset.from_pandas(pool_df, preserve_index=False)

        return DataLoader(
            pool_dataset,
            sampler=self.get_sampler(pool_dataset, RunningStage.POOL),
            collate_fn=self.get_collate_fn(RunningStage.POOL),
        )


class ActiveClassificationDataModule(ActiveDataModule):
    def __init__(
        self, tokenizer: Optional[PreTrainedTokenizerBase] = None, max_source_length: int = 128, **kwargs
    ) -> None:
        self._hparams_ignore.append("tokenizer")
        super().__init__(**kwargs)
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length

    def get_collate_fn(self, stage: Optional[str] = None) -> Optional[Callable]:

        return partial(
            collate_fn,
            max_source_length=self.max_source_length,
            columns_on_cpu=[SpecialColumns.ID] if stage == RunningStage.POOL else [],
            pad_token_id=self.tokenizer.pad_token_id,
            pad_fn=_pad,
        )
    
    @property
    def labels(self) -> List[str]:
        assert InputColumns.TARGET in self.train_dataset.features, KeyError(
            "A prepared dataset needs to have a `labels` column."
        )
        return self.train_dataset.features[InputColumns.TARGET].names

    @property
    def id2label(self) -> Dict[int, str]:
        return dict(enumerate(self.labels))

    @property
    def label2id(self) -> Dict[str, int]:
        return {v: k for k, v in self.id2label.items()}


"""
Define as globals otherwise pickle complains when running in multi-gpu
"""


def collate_fn(
    batch: List[Dict[str, Union[List[str], Tensor]]],
    max_source_length: int,
    columns_on_cpu: List[str],
    pad_token_id: int,
    pad_fn: Callable,
) -> Dict[str, Union[List[str], Tensor]]:
    # NOTE: beacuse of the batch_sampler we already obtain dict of lists
    # however the dataloader will try to create a list, so we have to unpack it
    assert len(batch) == 1, "Look at the data collator"
    batch = batch[0]

    # remove string columns that cannot be transfered on gpu
    columns_on_cpu = {col: batch.pop(col) for col in columns_on_cpu if col in batch}

    labels = batch.pop("labels", None)

    # input_ids and attention_mask to tensor
    # truncate -> convert to tensor -> pad
    batch = {
        k: pad_fn(
            inputs=batch[k],
            padding_value=pad_token_id,
            max_length=max_source_length,
        )
        for k in ("input_ids", "attention_mask")
    }

    if labels is not None:
        batch["labels"] = torch.tensor(labels, dtype=torch.long)

    # add things that need to remain on cpu
    if len(columns_on_cpu) > 0:
        batch["on_cpu"] = columns_on_cpu

    return batch
