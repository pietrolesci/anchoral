# Here we define DataModule that work with HuggingFace Datasets.
# We assume that each dataset is already processed and ready for training.
# Think of the DataModule is the last step of the data preparation pipeline.
#
#   download data -> (process data -> prepare data) -> datamodule -> model
#
# That is, the DataModule is only used to feed data to the model during training
# and evaluation.
# In addition, the ActiveDataModule also implements the logic to label data.
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from datasets import Dataset
from torch.utils.data import DataLoader

from src.energizer.data import DataModule
from src.energizer.enums import InputKeys, RunningStage, SpecialKeys


class ActiveDataModule(DataModule):
    _df: pd.DataFrame = None

    """
    Properties
    """

    @property
    def should_val_split(self) -> bool:
        """If user passed a validation dataset, we do not take validation indices from annotations."""
        return self.validation_dataset is None

    @property
    def total_data_size(self) -> int:
        return self.train_size + self.pool_size

    @property
    def train_size(self) -> int:
        return self._df[SpecialKeys.IS_LABELLED].sum()

    @property
    def pool_size(self) -> int:
        return len(self._df) - self.train_size

    @property
    def train_indices(self) -> np.ndarray:
        return self._df.loc[
            (self._df[SpecialKeys.IS_LABELLED] == True) & ((self._df[SpecialKeys.IS_VALIDATION] == False)),
            SpecialKeys.ID,
        ].values

    @property
    def pool_indices(self) -> np.ndarray:
        return self._df.loc[(self._df[SpecialKeys.IS_LABELLED] == False), SpecialKeys.ID].values

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
        return int(self._id[SpecialKeys.LABELLING_ROUND].max())

    @property
    def data_statistics(self) -> Dict[str, int]:
        return {
            "total_data_size": self.total_data_size,
            "train_size": self.train_size,
            "pool_size": self.pool_size,
            "num_train_batches": len(self.train_loader()),
            "num_pool_batches": len(self.pool_loader()),
        }

    """
    Helper methods
    """

    def setup(self, stage: Optional[str] = None) -> None:
        # with_format does not remove columns completely;
        # when the Dataset is cast to pandas they remain, so remove
        cols = list(self.train_dataset[0].keys())

        self._df = (
            self.train_dataset.to_pandas()
            .loc[:, cols]
            .assign(
                **{
                    SpecialKeys.IS_LABELLED: False,
                    SpecialKeys.IS_VALIDATION: False,
                    SpecialKeys.LABELLING_ROUND: -100,
                }
            )
        )

    def mask_train_from_index(self) -> None:
        if self.index is None:
            return
        train_ids = self.train_indices
        for i in train_ids:
            self.index.mark_deleted(i)

    def unmask_train_from_index(self) -> None:
        if self.index is None:
            return
        train_ids = self.train_indices
        for i in train_ids:
            self.index.unmark_deleted(i)

    def get_train_embeddings(self) -> np.ndarray:
        self.unmask_train_from_index()
        embeddings = self.index.get_items(self.train_indices)
        self.mask_train_from_index()
        return embeddings

    def get_pool_embeddings(self) -> np.ndarray:
        return self.index.get_items(self.pool_indices)
    
    def save_labelled_dataset(self, save_dir: str) -> None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        (
            self._df.loc[self._df[SpecialKeys.IS_LABELLED] == True].to_parquet(
                save_dir / "labelled_dataset.parquet", index=False
            )
        )

    """
    Main methods
    """

    def label(self, indices: List[int], round_idx: Optional[int] = None, val_perc: Optional[float] = None) -> None:
        """Moves instances at index `pool_idx` from the `pool_fold` to the `train_fold`.

        Args:
            pool_idx (List[int]): The index (relative to the pool_fold, not the overall data) to label.
        """
        assert isinstance(indices, list), ValueError(f"`indices` must be of type `List[int]`, not {type(indices)}.")
        assert isinstance(val_perc, float) or val_perc is None, ValueError(
            f"`val_perc` must be of type `float`, not {type(val_perc)}"
        )

        mask = self._df[SpecialKeys.ID].isin(indices)
        self._df.loc[mask, SpecialKeys.IS_LABELLED] = True
        self._df.loc[mask, SpecialKeys.LABELLING_ROUND] = round_idx

        if self.should_val_split and val_perc is not None:
            n_val = round(val_perc * len(indices)) or 1
            val_indices = self._rng.choice(indices, size=n_val, replace=False)
            self._df.loc[self._df[SpecialKeys.ID].isin(val_indices), SpecialKeys.IS_VALIDATION] = True

        # remove instance from the index
        if self._index is not None:
            for idx in indices:
                self.index.mark_deleted(idx)

    def set_initial_budget(self, budget: int, sampling: str = "random", val_perc: Optional[float] = None) -> None:
        assert sampling == "random", "Only `random` is supported by default. Write your own sampling."
        pool_df = self._df.loc[(self._df[SpecialKeys.IS_LABELLED] == False), [SpecialKeys.ID, InputKeys.TARGET]]
        indices = self._rng.choice(pool_df[SpecialKeys.ID], size=budget, replace=False).tolist()
        self.label(indices, round_idx=-1, val_perc=val_perc)


    """
    DataLoaders
    """

    def train_loader(self) -> DataLoader:
        if self.train_size > 0:
            train_df = self._df.loc[
                (self._df[SpecialKeys.IS_LABELLED] == True) & (self._df[SpecialKeys.IS_VALIDATION] == False)
            ]
            train_dataset = Dataset.from_pandas(train_df, preserve_index=False)

            return DataLoader(
                train_dataset,
                sampler=self.get_sampler(train_dataset, RunningStage.TRAIN),
                collate_fn=self.get_collate_fn(RunningStage.TRAIN),
            )

    def validation_loader(self) -> Optional[DataLoader]:                
        if self.should_val_split and self._df[SpecialKeys.IS_VALIDATION].sum() > 0:
            val_df = self._df.loc[
                (self._df[SpecialKeys.IS_LABELLED] == True) & (self._df[SpecialKeys.IS_VALIDATION] == True)
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

    def pool_loader(self, subset_indices: Optional[List[int]] = None) -> DataLoader:
        pool_df = self._df.loc[
            (self._df[SpecialKeys.IS_LABELLED] == False),
            [i for i in self._df.columns if i != InputKeys.TARGET],
        ]

        if subset_indices is not None:
            pool_df = pool_df.loc[pool_df[SpecialKeys.ID].isin(subset_indices)]

        # for performance reasons
        pool_dataset = Dataset.from_pandas(
            df=(
                pool_df
                .assign(length=lambda df_: df_[InputKeys.INPUT_IDS].map(len))
                .sort_values("length")
                .drop(columns=["length"])
            ),
            preserve_index=False,
        )

        return DataLoader(
            pool_dataset,
            sampler=self.get_sampler(pool_dataset, RunningStage.POOL),
            collate_fn=self.get_collate_fn(RunningStage.POOL),
        )

