import os
from functools import partial
from typing import Callable, Dict, List, Optional, Union

import torch
from datasets import DatasetDict
from torch import Tensor

from src.data.datamodule import DataModule, _pad
from src.enums import InputKeys, RunningStage, SpecialKeys
from transformers import PreTrainedTokenizerBase

"""
A very tailored datamodule for HuggingFace datasets
"""


class ClassificationDataModule(DataModule):
    _default_columns: List[str] = [InputKeys.TARGET, InputKeys.INPUT_IDS, InputKeys.ATT_MASK]

    def __init__(self, tokenizer: Optional[PreTrainedTokenizerBase], max_source_length: int = 128, **kwargs) -> None:
        self._hparams_ignore.append("tokenizer")
        super().__init__(**kwargs)
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        self._tokenizer = tokenizer
        self.max_source_length = max_source_length

    @property
    def tokenizer(self) -> PreTrainedTokenizerBase:
        return self._tokenizer

    @property
    def labels(self) -> List[str]:
        assert InputKeys.TARGET in self.train_dataset.features, KeyError(
            "A prepared dataset needs to have a `labels` column."
        )
        return self.train_dataset.features[InputKeys.TARGET].names

    @property
    def id2label(self) -> Dict[int, str]:
        return dict(enumerate(self.labels))

    @property
    def label2id(self) -> Dict[str, int]:
        return {v: k for k, v in self.id2label.items()}

    def get_collate_fn(self, stage: Optional[str] = None) -> Optional[Callable]:
        return partial(
            collate_fn,
            columns_on_cpu=self.columns_on_cpu,
            max_source_length=self.max_source_length,
            pad_token_id=self.tokenizer.pad_token_id,
            pad_fn=_pad,
        )

    @classmethod
    def from_dataset_dict(
        cls,
        dataset_dict: DatasetDict,
        tokenizer: PreTrainedTokenizerBase,
        columns_to_keep: Optional[List[str]] = None,
        columns_on_cpu: Optional[List[str]] = None,
        **kwargs,
    ) -> None:
        columns_to_keep = columns_to_keep or []
        columns_to_keep += cls._default_columns

        columns_on_cpu = columns_on_cpu or []
        assert all(
            col in columns_to_keep for col in columns_on_cpu
        ), "`columns_on_cpu` must be a subset of `columns_to_keep`"

        datasets = {}
        for stage in RunningStage:
            if stage in dataset_dict:
                dataset = dataset_dict[stage]

                if SpecialKeys.ID in dataset.features and SpecialKeys.ID not in columns_to_keep:
                    columns_to_keep.append(SpecialKeys.ID)
                    columns_on_cpu.append(SpecialKeys.ID)

                datasets[f"{stage}_dataset"] = dataset.with_format(columns=list(set(columns_to_keep)))

        cls.columns_on_cpu = list(set(columns_on_cpu))

        return cls(**datasets, tokenizer=tokenizer, **kwargs)


def collate_fn(
    batch: List[Dict[str, Union[List[str], Tensor]]],
    columns_on_cpu: List[str],
    max_source_length: int,
    pad_token_id: int,
    pad_fn: Callable,
) -> Dict[str, Union[List[str], Tensor]]:
    # NOTE: beacuse of the batch_sampler we already obtain dict of lists
    # however the dataloader will try to create a list, so we have to unpack it
    assert len(batch) == 1, "Look at the data collator"
    batch = batch[0]

    # remove string columns that cannot be transfered on gpu
    columns_on_cpu = {col: batch.pop(col) for col in columns_on_cpu if col in batch}

    labels = batch.pop(InputKeys.TARGET, None)

    # input_ids and attention_mask to tensor
    # truncate -> convert to tensor -> pad
    batch = {
        k: pad_fn(
            inputs=batch[k],
            padding_value=pad_token_id,
            max_length=max_source_length,
        )
        for k in (InputKeys.INPUT_IDS, InputKeys.ATT_MASK)
    }

    if labels is not None:
        batch[InputKeys.TARGET] = torch.tensor(labels, dtype=torch.long)

    # add things that need to remain on cpu
    if len(columns_on_cpu) > 0:
        batch[InputKeys.ON_CPU] = columns_on_cpu

    return batch
