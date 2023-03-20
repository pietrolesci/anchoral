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
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.utils import resample
from torch.utils.data import DataLoader

from src.energizer.data import DataModule
from src.energizer.enums import InputKeys, RunningStage, SpecialKeys


class ActiveDataModule(DataModule):
    _df: pd.DataFrame = None

    """
    Properties
    """

    def _resolve_round(self, round: Optional[int] = None) -> Union[float, int]:
        if round is None:
            return float("Inf")
        return round

    def _labelled_mask(self, round: Optional[int] = None) -> pd.Series:
        return (self._df[SpecialKeys.IS_LABELLED] == True) & (
            self._df[SpecialKeys.LABELLING_ROUND] <= self._resolve_round(round)
        )

    def _train_mask(self, round: Optional[int] = None) -> pd.Series:
        return self._labelled_mask(round) & (self._df[SpecialKeys.IS_VALIDATION] == False)

    def _validation_mask(self, round: Optional[int] = None) -> pd.Series:
        return self._labelled_mask(round) & (self._df[SpecialKeys.IS_VALIDATION] == True)

    def _pool_mask(self, round: Optional[int] = None) -> pd.Series:
        return (self._df[SpecialKeys.IS_LABELLED] == False) | (
            self._df[SpecialKeys.LABELLING_ROUND] > self._resolve_round(round)
        )

    @property
    def test_size(self) -> int:
        if self.test_dataset is None:
            return 0
        return len(self.test_dataset)

    @property
    def has_test_data(self) -> bool:
        return self.test_size > 0

    @property
    def last_labelling_round(self) -> int:
        """Returns the number of the last active learning step."""
        return int(self._df[SpecialKeys.LABELLING_ROUND].max())

    @property
    def query_size(self) -> int:
        last = len(self._df.loc[self._df[SpecialKeys.LABELLING_ROUND] <= self.last_labelling_round])
        prev = len(self._df.loc[self._df[SpecialKeys.LABELLING_ROUND] <= self.last_labelling_round - 1])
        return last - prev

    @property
    def initial_budget(self) -> int:
        return (self._df[SpecialKeys.LABELLING_ROUND] == 0).sum()

    def train_size(self, round: Optional[int] = None) -> int:
        return self._train_mask(round).sum()

    def has_train_data(self, round: Optional[int] = None) -> bool:
        return self.train_size(round) > 0

    def validation_size(self, round: Optional[int] = None) -> int:
        return self._validation_mask(round).sum()

    def has_validation_data(self, round: Optional[int] = None) -> bool:
        return self.validation_size(round) > 0

    def total_labelled_size(self, round: Optional[int] = None) -> int:
        return self._labelled_mask(round).sum()

    def pool_size(self, round: Optional[int] = None) -> int:
        return self._pool_mask(round).sum()

    def has_unlabelled_data(self, round: Optional[int] = None) -> bool:
        return self.pool_size(round) > 0

    def train_indices(self, round: Optional[int] = None) -> np.ndarray:
        return self._df.loc[self._train_mask(round), SpecialKeys.ID].values

    def validation_indices(self, round: Optional[int] = None) -> np.ndarray:
        return self._df.loc[self._validation_mask(round), SpecialKeys.ID].values

    def pool_indices(self, round: Optional[int] = None) -> np.ndarray:
        return self._df.loc[self._pool_mask(round), SpecialKeys.ID].values

    def data_statistics(self, round: Optional[int] = None) -> Dict[str, int]:
        return {
            "train_size": self.train_size(round),
            "validation_size": self.validation_size(round),
            "test_size": self.test_size,
            "pool_size": self.pool_size(round),
            "total_labelled_size": self.total_labelled_size(round),
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

    def get_labelled_dataset(self) -> pd.DataFrame:
        cols = [i for i in SpecialKeys] + [InputKeys.TARGET]
        return self._df.loc[self._df[SpecialKeys.IS_LABELLED] == True, cols]

    def set_labelled_dataset(self, df: pd.DataFrame) -> None:
        assert df[SpecialKeys.ID].isin(self._df[SpecialKeys.ID]).all()

        to_keep = self._df.loc[~self._df[SpecialKeys.ID].isin(df[SpecialKeys.ID])]

        cols = [SpecialKeys.IS_LABELLED, SpecialKeys.IS_VALIDATION, SpecialKeys.LABELLING_ROUND]
        to_update = pd.merge(df[cols + [SpecialKeys.ID]], self._df.drop(columns=cols), on=SpecialKeys.ID, how="inner")
        assert not to_update[SpecialKeys.ID].isin(to_keep[SpecialKeys.ID]).all()

        new_df = pd.concat([to_keep, to_update])
        assert new_df.shape == self._df.shape
        assert df[SpecialKeys.IS_LABELLED].sum() == new_df[SpecialKeys.IS_LABELLED].sum()
        assert df[SpecialKeys.IS_VALIDATION].sum() == new_df[SpecialKeys.IS_VALIDATION].sum()

        self._df = new_df.copy()

    def save_labelled_dataset(self, save_dir: str) -> None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        self.get_labelled_dataset().to_parquet(save_dir / "labelled_dataset.parquet", index=False)

    """
    Main methods
    """

    def label(
        self,
        indices: List[int],
        round_idx: Optional[int] = None,
        validation_perc: Optional[float] = None,
        validation_sampling: Optional[str] = None,
    ) -> int:
        """Moves instances at index `pool_idx` from the `pool_fold` to the `train_fold`.

        Args:
            pool_idx (List[int]): The index (relative to the pool_fold, not the overall data) to label.
        """
        assert isinstance(indices, list), ValueError(f"`indices` must be of type `List[int]`, not {type(indices)}.")
        assert isinstance(validation_perc, float) or validation_perc is None, ValueError(
            f"`validation_perc` must be of type `float`, not {type(validation_perc)}"
        )

        mask = self._df[SpecialKeys.ID].isin(indices)
        self._df.loc[mask, SpecialKeys.IS_LABELLED] = True
        self._df.loc[mask, SpecialKeys.LABELLING_ROUND] = round_idx

        if validation_perc is not None and validation_perc > 0.0:
            n_val = round(validation_perc * len(indices)) or 1  # at least add one
            current_df = self._df.loc[mask, [SpecialKeys.ID, InputKeys.TARGET]]
            val_indices = self.sample(
                indices=current_df[SpecialKeys.ID].tolist(),
                size=n_val,
                labels=current_df[InputKeys.TARGET],
                sampling=validation_sampling,
            )
            self._df.loc[self._df[SpecialKeys.ID].isin(val_indices), SpecialKeys.IS_VALIDATION] = True

        # remove instance from the index
        if self._index is not None:
            for idx in indices:
                self.index.mark_deleted(idx)

        return mask.sum()

    def set_initial_budget(
        self,
        budget: int,
        validation_perc: Optional[float] = None,
        sampling: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> None:
        df = self._df.loc[(self._df[SpecialKeys.IS_LABELLED] == False), [SpecialKeys.ID, InputKeys.TARGET]]
        # sample from the pool
        indices = self.sample(
            indices=df[SpecialKeys.ID].tolist(),
            size=budget,
            labels=df[InputKeys.TARGET].tolist(),
            sampling=sampling,
            seed=seed,
        )

        # actually label
        self.label(indices=indices, round_idx=0, validation_perc=validation_perc, validation_sampling=sampling)

    def sample(
        self,
        indices: List[int],
        size: int,
        labels: Optional[List[int]],
        sampling: Optional[str] = None,
        # seed: Optional[int] = None,
    ) -> List[int]:
        """Makes sure to seed everything consistently."""
        # _rng = check_random_state(seed)

        if sampling is None or sampling == "random":
            sample = self._rng.choice(indices, size=size, replace=False)

        elif sampling == "stratified" and labels is not None:
            sample = resample(
                indices,
                replace=False,
                stratify=labels,
                n_samples=size,
                random_state=self._rng,
            )

        else:
            raise ValueError("Only `random` and `stratified` are supported by default.")

        assert len(set(sample)) == size

        return sample

    """
    DataLoaders
    """

    def train_loader(self, round: Optional[int] = None) -> Optional[DataLoader]:
        round = self._resolve_round(round)
        if self.train_size(round) > 0:
            df = self._df.loc[self._train_mask(round)]
            dataset = Dataset.from_pandas(df, preserve_index=False)

            return self.get_loader(RunningStage.TRAIN, dataset)

    def validation_loader(self, round: Optional[int] = None) -> Optional[DataLoader]:
        round = self._resolve_round(round)
        if self.validation_size(round) > 0:
            df = self._df.loc[self._validation_mask(round)]
            dataset = Dataset.from_pandas(df, preserve_index=False)
            return self.get_loader(RunningStage.VALIDATION, dataset)

    def pool_loader(self, subset_indices: Optional[List[int]] = None) -> DataLoader:
        df = self._df.loc[
            (self._df[SpecialKeys.IS_LABELLED] == False),
            [i for i in self._df.columns if i != InputKeys.TARGET],
        ]

        if subset_indices is not None:
            df = df.loc[df[SpecialKeys.ID].isin(subset_indices)]

        # for performance reasons
        dataset = Dataset.from_pandas(
            df=(
                df.assign(length=lambda df_: df_[InputKeys.INPUT_IDS].map(len))
                .sort_values("length")
                .drop(columns=["length"])
            ),
            preserve_index=False,
        )

        return self.get_loader(RunningStage.POOL, dataset)
