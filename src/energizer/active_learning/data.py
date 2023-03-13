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
from sklearn.utils import resample
from sklearn.utils.validation import check_random_state
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
    def train_size(self) -> int:
        return ((self._df[SpecialKeys.IS_LABELLED] == True) & (self._df[SpecialKeys.IS_VALIDATION] == False)).sum()

    @property
    def validation_size(self) -> int:
        if self.should_val_split:
            return ((self._df[SpecialKeys.IS_LABELLED] == True) & (self._df[SpecialKeys.IS_VALIDATION] == True)).sum()
        return len(self.validation_dataset)

    @property
    def total_labelled_size(self) -> int:
        return self.train_size + self.validation_size

    @property
    def test_size(self) -> Optional[int]:
        if self.test_dataset is not None:
            return len(self.test_dataset)

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
            "train_size": self.train_size,
            "validation_size": self.validation_size,
            "test_size": self.test_size,
            "pool_size": self.pool_size,
            "total_labelled_size": self.total_labelled_size,
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

        # check consistency of the index
        assert self._df[SpecialKeys.ID].nunique() == len(self._df)

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
        cols = [i for i in SpecialKeys] + [InputKeys.TARGET]
        (
            self._df.loc[self._df[SpecialKeys.IS_LABELLED] == True, cols].to_parquet(
                save_dir / "labelled_dataset.parquet", index=False
            )
        )

    """
    Main methods
    """

    def label(
        self,
        indices: List[int],
        round_idx: Optional[int] = None,
        val_perc: Optional[float] = None,
        val_sampling: Optional[str] = None,
    ) -> None:
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
            n_val = round(val_perc * len(indices)) or 1  # at least add one
            current_df = self._df.loc[mask, [SpecialKeys.ID, InputKeys.TARGET]]
            val_indices = self.sample(
                indices=current_df[SpecialKeys.ID].tolist(),
                size=n_val,
                labels=current_df[InputKeys.TARGET],
                sampling=val_sampling,
            )
            self._df.loc[self._df[SpecialKeys.ID].isin(val_indices), SpecialKeys.IS_VALIDATION] = True

        # remove instance from the index
        if self._index is not None:
            for idx in indices:
                self.index.mark_deleted(idx)

    def set_initial_budget(
        self, budget: int, val_perc: Optional[float] = None, sampling: Optional[str] = None, seed: Optional[int] = None
    ) -> None:
        pool_df = self._df.loc[(self._df[SpecialKeys.IS_LABELLED] == False), [SpecialKeys.ID, InputKeys.TARGET]]
        # sample from the pool
        indices = self.sample(
            indices=pool_df[SpecialKeys.ID].tolist(),
            size=budget,
            labels=pool_df[InputKeys.TARGET].tolist(),
            sampling=sampling,
            seed=seed,
        )

        # actually label
        self.label(indices=indices, round_idx=-1, val_perc=val_perc, val_sampling=sampling)

    def sample(
        self,
        indices: List[int],
        size: int,
        labels: Optional[List[int]],
        sampling: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> List[int]:
        """Makes sure to seed everything consistently."""
        _rng = check_random_state(seed)

        if sampling is None or sampling == "random":
            sample = _rng.choice(indices, size=size, replace=False)

        elif sampling == "stratified" and labels is not None:
            sample = resample(
                indices,
                replace=False,
                stratify=labels,
                n_samples=size,
                random_state=_rng,
            )

        else:
            raise ValueError("Only `random` and `stratified` are supported by default.")

        assert len(set(sample)) == size

        return sample

    """
    DataLoaders
    """

    def train_loader(self) -> DataLoader:
        if self.train_size > 0:
            train_df = self._df.loc[
                (self._df[SpecialKeys.IS_LABELLED] == True) & (self._df[SpecialKeys.IS_VALIDATION] == False)
            ]

            self.train_dataset = Dataset.from_pandas(train_df, preserve_index=False)

            return self.get_loader(RunningStage.TRAIN)

    def validation_loader(self) -> Optional[DataLoader]:
        if self.should_val_split and self._df[SpecialKeys.IS_VALIDATION].sum() > 0:
            val_df = self._df.loc[
                (self._df[SpecialKeys.IS_LABELLED] == True) & (self._df[SpecialKeys.IS_VALIDATION] == True)
            ]
            val_dataset = Dataset.from_pandas(val_df, preserve_index=False)
        else:
            val_dataset = self.validation_dataset

        if val_dataset is not None and len(val_dataset) > 0:
            self.validation_dataset = val_dataset
            return self.get_loader(RunningStage.VALIDATION)

    def pool_loader(self, subset_indices: Optional[List[int]] = None) -> DataLoader:
        pool_df = self._df.loc[
            (self._df[SpecialKeys.IS_LABELLED] == False),
            [i for i in self._df.columns if i != InputKeys.TARGET],
        ]

        if subset_indices is not None:
            pool_df = pool_df.loc[pool_df[SpecialKeys.ID].isin(subset_indices)]

        # for performance reasons
        self.pool_dataset = Dataset.from_pandas(
            df=(
                pool_df.assign(length=lambda df_: df_[InputKeys.INPUT_IDS].map(len))
                .sort_values("length")
                .drop(columns=["length"])
            ),
            preserve_index=False,
        )

        return self.get_loader(RunningStage.POOL)
