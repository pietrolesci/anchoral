from abc import ABC, abstractmethod
from typing import Literal, Optional, Union

import numpy as np
import pandas as pd
from lightning.fabric.wrappers import _FabricModule
from scipy.special import softmax
from sklearn.utils import check_random_state

from energizer.active_learning.datastores.base import ActivePandasDataStoreWithIndex
from energizer.active_learning.strategies.diversity import EmbeddingClustering
from energizer.active_learning.strategies.random import RandomStrategy
from energizer.active_learning.strategies.two_stage import BaseSubsetWithSearchStrategy
from energizer.active_learning.strategies.uncertainty import UncertaintyBasedStrategy
from energizer.enums import InputKeys, SpecialKeys


class AnchorStrategy(ABC):
    @abstractmethod
    def get_anchors(
        self, model: _FabricModule, datastore: ActivePandasDataStoreWithIndex, num_anchors: int, class_ids: list[int]
    ) -> list[int]: ...


class RandomAnchorStrategy(AnchorStrategy, RandomStrategy):
    def get_anchors(
        self, model: _FabricModule, datastore: ActivePandasDataStoreWithIndex, num_anchors: int, class_ids: list[int]
    ) -> list[int]:
        return self.rng.choice(class_ids, size=min(num_anchors, len(class_ids)), replace=False).tolist()


class ClusteringAnchorStrategy(AnchorStrategy, EmbeddingClustering):
    def get_anchors(
        self, model: _FabricModule, datastore: ActivePandasDataStoreWithIndex, num_anchors: int, class_ids: list[int]
    ) -> list[int]:
        if len(class_ids) <= num_anchors:
            return class_ids
        embs, ids = self.get_embeddings_and_ids(model, datastore, num_anchors, class_ids=class_ids)
        return self.select_from_embeddings(model, datastore, num_anchors, embs, ids)  # type: ignore

    def get_embeddings_and_ids(
        self, model: _FabricModule, datastore: ActivePandasDataStoreWithIndex, query_size: int, **kwargs
    ) -> tuple[np.ndarray, np.ndarray]:
        ids = kwargs["class_ids"]
        return datastore.get_train_embeddings(ids), np.array(ids)


class UncertaintyAnchorStrategy(AnchorStrategy, UncertaintyBasedStrategy):
    def get_anchors(
        self, model: _FabricModule, datastore: ActivePandasDataStoreWithIndex, num_anchors: int, class_ids: list[int]
    ) -> list[int]:
        train_loader = self.get_train_loader(datastore, with_indices=class_ids)  # type: ignore
        if train_loader is None or len(train_loader.dataset or []) <= num_anchors:  # type: ignore
            return class_ids
        return self.compute_most_uncertain(model, train_loader, num_anchors)


class AnchorAL(BaseSubsetWithSearchStrategy):
    _num_anchors: int

    def __init__(
        self,
        *args,
        num_anchors: int,
        anchor_strategy_minority: Union[Literal["all"], AnchorStrategy],
        anchor_strategy_majority: Union[Literal["all"], AnchorStrategy],
        minority_classes_ids: Optional[list[int]] = None,
        num_neighbours: int,
        max_search_size: Optional[int] = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, num_neighbours=num_neighbours, max_search_size=max_search_size, **kwargs)
        self._num_anchors = num_anchors
        self.anchor_strategy_majority = anchor_strategy_majority
        self.anchor_strategy_minority = anchor_strategy_minority
        self.minority_classes_ids = minority_classes_ids or []

    @property
    def num_anchors(self) -> int:
        return self._num_anchors

    def select_search_query(
        self, model: _FabricModule, datastore: ActivePandasDataStoreWithIndex, query_size: int, **kwargs
    ) -> list[int]:  # Dict[str, List[int]]:
        train_df = datastore.get_by_ids(datastore.get_train_ids())[[SpecialKeys.ID, InputKeys.LABELS]]
        if len(train_df) == 0 or (self.num_anchors > 0 and len(train_df) < self.num_anchors):
            return train_df[SpecialKeys.ID].tolist()

        anchor_ids = []
        classes = sorted(train_df[InputKeys.LABELS].unique().tolist())  # sort to avoid any randomness
        for c in classes:
            anchor_strategy = (
                self.anchor_strategy_minority if c in self.minority_classes_ids else self.anchor_strategy_majority
            )
            ids = train_df.loc[train_df[InputKeys.LABELS] == c, SpecialKeys.ID].tolist()  # type: ignore

            if len(ids) == 0:
                continue

            elif anchor_strategy == "all" or len(ids) < self.num_anchors:
                _ids = ids

            else:
                # assert isinstance(anchor_strategy, AnchorStrategy)
                _ids = anchor_strategy.get_anchors(model, datastore, self.num_anchors, ids)

            assert len(_ids) == len(set(_ids)), f"{_ids}"  # if we find duplicates, there are bugs
            anchor_ids += _ids

        return anchor_ids

    def get_query_embeddings(
        self, datastore: ActivePandasDataStoreWithIndex, search_query_ids: list[int]
    ) -> np.ndarray:
        return datastore.get_train_embeddings(search_query_ids)

    def get_subpool_ids_from_search_results(
        self, candidate_df: pd.DataFrame, datastore: ActivePandasDataStoreWithIndex
    ) -> list[int]:
        # aggregate by pool instances
        df = (
            candidate_df.groupby(SpecialKeys.ID)
            .agg(dists=("dists", "mean"))
            .reset_index()
            # convert cosine distance in similarity (i.e., higher means more similar)
            .assign(scores=lambda _df: 1 - _df["dists"])
        )
        # if self.aggregation == "weighted":
        #     df = self.weighted_aggregate_scores(candidate_df, datastore)
        # else:
        #     df = self.unweighted_aggregate_scores(candidate_df, datastore)

        # select topk
        indices = df.sort_values("scores", ascending=False).head(min(self.subpool_size, len(df)))[SpecialKeys.ID]
        return indices.tolist()


class AnchorALWithSampling(AnchorAL):
    def __init__(self, *args, seed: int, **kwargs) -> None:
        super().__init__(*args, seed=seed, **kwargs)
        self.gumbel_rng = check_random_state(seed)

    def get_subpool_ids_from_search_results(
        self, candidate_df: pd.DataFrame, datastore: ActivePandasDataStoreWithIndex
    ) -> list[int]:
        # train_df = datastore.get_by_ids(candidate_df["search_query_uid"].tolist())[[SpecialKeys.ID, InputKeys.LABELS]]
        # minority_ids = train_df.loc[train_df[InputKeys.LABELS].isin(self.minority_classes_ids), SpecialKeys.ID].tolist()
        # candidate_df = candidate_df.loc[candidate_df[SpecialKeys.ID].isin(minority_ids)]

        df = (
            candidate_df.groupby(SpecialKeys.ID)
            .agg(dists=("dists", "mean"))
            .reset_index()
            # convert cosine distance in similarity (i.e., higher means more similar)
            .assign(scores=lambda _df: 1 - _df["dists"])
        )

        probs = softmax(df["scores"], axis=0)
        indices = self.gumbel_rng.choice(
            df[SpecialKeys.ID].values, size=min(self.subpool_size, len(df)), replace=False, p=probs
        )
        return indices.tolist()

        # convert cosine distance in similarity (i.e., higher means more similar)
        # df["perturbed_scores"] = df["scores"] + self.gumbel_rng.gumbel(0.0, 1.0, size=len(df))

        # select topk
        # indices = df.sort_values("perturbed_scores", ascending=False).head(min(self.subpool_size, len(df)))[
        #     SpecialKeys.ID
        # ]
        # return indices.tolist()
