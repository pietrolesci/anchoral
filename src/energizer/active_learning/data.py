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
from torch.utils.data import DataLoader

from src.energizer.data import DataModule
from src.energizer.enums import InputKeys, RunningStage, SpecialKeys


class ActiveDataModule(DataModule):
    _df: pd.DataFrame = None

    """
    Properties
    """

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
        return (self._df[SpecialKeys.LABELLING_ROUND] == -1).sum()
    
    def train_size(self, round: Optional[int] = None) -> int:
        round = round or float("Inf")
        return (
            (self._df[SpecialKeys.IS_LABELLED] == True) 
            & (self._df[SpecialKeys.IS_VALIDATION] == False)
            & (self._df[SpecialKeys.LABELLING_ROUND] < round)
        ).sum()

    def has_train_data(self, round: Optional[int] = None) -> bool:
        return self.train_size(round or float("Inf")) > 0

    def validation_size(self, round: Optional[int] = None) -> int:
        round = round or float("Inf")
        return (
            (self._df[SpecialKeys.IS_LABELLED] == True) 
            & (self._df[SpecialKeys.IS_VALIDATION] == True)
            & (self._df[SpecialKeys.LABELLING_ROUND] < round)
        ).sum()
    
    def has_validation_data(self, round: Optional[int] = None) -> bool:
        return self.validation_size(round or float("Inf")) > 0
    
    def total_labelled_size(self, round: Optional[int] = None) -> int:
        round = round or float("Inf")
        return self.train_size(round) + self.validation_size(round)

    def pool_size(self, round: Optional[int] = None) -> int:
        round = round or float("Inf")
        return (
            (self._df[SpecialKeys.IS_LABELLED] == False) 
            & (self._df[SpecialKeys.LABELLING_ROUND] < round)
        ).sum()

    def train_indices(self, round: Optional[int] = None) -> np.ndarray:
        round = round or float("Inf")
        return self._df.loc[
            (
                (self._df[SpecialKeys.IS_LABELLED] == True) 
                & (self._df[SpecialKeys.IS_VALIDATION] == False) 
                & (self._df[SpecialKeys.LABELLING_ROUND] < round),
            ),
            SpecialKeys.ID,
        ].values
   
    def pool_indices(self, round: Optional[int] = None) -> np.ndarray:
        round = round or float("Inf")
        return self._df.loc[
            (
                (self._df[SpecialKeys.IS_LABELLED] == False) 
                & (self._df[SpecialKeys.LABELLING_ROUND] < round),
            ),
            SpecialKeys.ID,
        ].values

    def has_unlabelled_data(self, round: Optional[int] = None) -> bool:
        """Checks whether there are data to be labelled."""
        return self.pool_size(round = round or float("Inf")) > 0

    def data_statistics(self, round: Optional[int] = None) -> Dict[str, int]:
        round = round or float("Inf")
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

    def save_labelled_dataset(self, save_dir: str) -> None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        self.get_labelled_dataset().to_parquet(
            save_dir / "labelled_dataset.parquet", index=False
        )

    """
    Main methods
    """

    def label(
        self,
        indices: List[int],
        round_idx: Optional[int] = None,
        validation_perc: Optional[float] = None,
        validation_sampling: Optional[str] = None,
    ) -> None:
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

        if validation_perc is not None:
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
        self.label(indices=indices, round_idx=-1, validation_perc=validation_perc, validation_sampling=sampling)

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
        round = round or float("Inf")
        if self.train_size(round) > 0:
            df = self._df.loc[
                (self._df[SpecialKeys.IS_LABELLED] == True) & (self._df[SpecialKeys.IS_VALIDATION] == False)
            ]
            if round is not None:
                df = df.loc[df[SpecialKeys.LABELLING_ROUND] < round]

            dataset = Dataset.from_pandas(df, preserve_index=False)

            return self.get_loader(RunningStage.TRAIN, dataset)

    def validation_loader(self, round: Optional[int] = None) -> Optional[DataLoader]:
        round = round or float("Inf")
        if self.validation_size(round) > 0:
            df = self._df.loc[
                (self._df[SpecialKeys.IS_LABELLED] == True) & (self._df[SpecialKeys.IS_VALIDATION] == True)
            ]
            if round is not None:
                df = df.loc[df[SpecialKeys.LABELLING_ROUND] < round]

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
