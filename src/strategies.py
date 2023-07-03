import math
import time
from typing import List, Optional

import numpy as np
import pandas as pd
from lightning.fabric.wrappers import _FabricModule
from numpy.random import RandomState
from scipy.special import softmax
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from sklearn.utils import check_random_state

from energizer.datastores import PandasDataStoreForSequenceClassification
from energizer.enums import InputKeys, SpecialKeys
from energizer.strategies import RandomStrategy as _RandomStrategy
from energizer.strategies import UncertaintyBasedStrategy as _UncertaintyBasedStrategy
from src.estimators import SequenceClassificationMixin


def run_kmeans(
    datastore: PandasDataStoreForSequenceClassification, ids: List[int], num_clusters: int, rng: RandomState, from_pool: bool = False,
) -> List[int]:
    if from_pool:
        embeddings = datastore.get_pool_embeddings(ids)
    else:
        embeddings = datastore.get_train_embeddings(ids)
    embeddings: np.ndarray = normalize(embeddings, axis=1)  # type: ignore

    num_clusters = min(embeddings.shape[0], num_clusters)

    cluster_learner = KMeans(n_clusters=num_clusters, n_init="auto", random_state=rng)
    cluster_learner.fit(embeddings)
    cluster_idxs = cluster_learner.predict(embeddings)

    # pick instances closest to the cluster centers
    centers = cluster_learner.cluster_centers_[cluster_idxs]
    dists = (embeddings - centers) ** 2
    dists = dists.sum(axis=1)
    closest_ids = [
        np.arange(embeddings.shape[0])[cluster_idxs == i][dists[cluster_idxs == i].argmin()].item()
        for i in range(num_clusters)
    ]
    return np.array(ids)[closest_ids].tolist()


class RandomStrategy(SequenceClassificationMixin, _RandomStrategy):
    ...


class UncertaintyStrategy(SequenceClassificationMixin, _UncertaintyBasedStrategy):
    ...


class UncertaintyMixin:
    def select(
        self,
        model: _FabricModule,
        datastore: PandasDataStoreForSequenceClassification,
        subset_ids: List[int],
        query_size: int,
    ) -> List[int]:
        """Runs uncertainty sampling using a subset of the pool.


        Returns:
            List[int]: The uids of the selected data points to annotate.
        """
        if len(subset_ids) <= query_size:
            return subset_ids
        pool_loader = self.configure_dataloader(datastore.pool_loader(with_indices=subset_ids))  # type: ignore
        self.progress_tracker.pool_tracker.max = len(pool_loader)  # type: ignore
        return self.compute_most_uncertain(model, pool_loader, query_size)  # type: ignore


class RandomSubset(SequenceClassificationMixin, UncertaintyMixin, _UncertaintyBasedStrategy):
    def __init__(self, *args, subpool_size: int, seed: int, **kwargs) -> None:
        """Strategy that runs uncertainty sampling on a random subset of the pool.

        Args:
            subpool_size (int): Size of the subset.
            seed (int): Random seed for the subset selection.
        """
        super().__init__(*args, **kwargs)
        self.subpool_size = subpool_size
        self.seed = seed
        self.rng = check_random_state(seed)

    def run_query(
        self, model: _FabricModule, datastore: PandasDataStoreForSequenceClassification, query_size: int
    ) -> List[int]:
        subpool_size = min(datastore.pool_size(), self.subpool_size)
        subpool_ids = datastore.sample_from_pool(size=subpool_size, mode="uniform", random_state=self.rng)
        self.log("summary/subpool_size", len(subpool_ids), step=self.progress_tracker.global_round)
        return self.select(model, datastore, subpool_ids, query_size)


class Tyrogue(RandomSubset):
    def __init__(self, *args, r: int, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.r = r

    def run_query(
        self, model: _FabricModule, datastore: PandasDataStoreForSequenceClassification, query_size: int
    ) -> List[int]:
        subpool_ids = datastore.sample_from_pool(
            size=min(datastore.pool_size(), self.subpool_size), mode="uniform", random_state=self.rng
        )
        num_clusters = query_size * self.r
        subpool_ids = run_kmeans(datastore, subpool_ids, num_clusters=num_clusters, rng=self.rng, from_pool=True)
        self.log("summary/subpool_size", len(subpool_ids), step=self.progress_tracker.global_round)
        return self.select(model, datastore, subpool_ids, query_size)


class BaseSubsetWithSearch(SequenceClassificationMixin, UncertaintyMixin, _UncertaintyBasedStrategy):
    _reason_df: pd.DataFrame = pd.DataFrame()

    def __init__(self, *args, num_neighbours: int, seed: int, max_search_size: Optional[int] = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.seed = seed
        self.pool_rng = check_random_state(seed)
        self.num_neighbours = num_neighbours
        self.max_search_size = max_search_size

    def run_query(
        self, model: _FabricModule, datastore: PandasDataStoreForSequenceClassification, query_size: int
    ) -> List[int]:

        # GET ANCHORS
        train_ids = self.get_anchors(model, datastore)
        # self.log("summary/num_anchors", len(train_ids), step=self.progress_tracker.global_round)

        if len(train_ids) == 0:
            # if cold-starting there is no training embedding, fall-back to random sampling
            return datastore.sample_from_pool(size=query_size, mode="uniform", random_state=self.pool_rng)

        # SEARCH ANCHORS
        train_embeddings = datastore.get_train_embeddings(train_ids)
        candidate_df = self.search_pool(datastore, train_embeddings, train_ids)

        # USE RESULTS TO SUBSET POOL
        subpool_ids = self.get_subpool_ids(candidate_df, datastore)
        self.log("summary/subpool_size", len(subpool_ids), step=self.progress_tracker.global_round)

        # RUN ACTIVE LEARNING CRITERION
        selected_ids = self.select(model, datastore, subpool_ids, query_size)

        # add traceability into the datastore
        self._record_reason(datastore, candidate_df, selected_ids)

        return selected_ids

    def get_anchors(self, model: _FabricModule, datastore: PandasDataStoreForSequenceClassification) -> List[int]:
        raise NotImplementedError

    def search_pool(
        self,
        datastore: PandasDataStoreForSequenceClassification,
        query: np.ndarray,
        train_ids: List[int],
    ) -> pd.DataFrame:
        """Get neighbours of training instances from the pool."""

        num_neighbours = self.num_neighbours
        if self.max_search_size is not None:
            # NOTE: this can create noise in the experimentss
            num_neighbours = min(self.num_neighbours, math.floor(self.max_search_size / query.shape[0]))

        start_time = time.perf_counter()
        ids, dists = datastore.search(query=query, query_size=num_neighbours, query_in_set=False)
        elapsed = time.perf_counter() - start_time

        candidate_df = pd.DataFrame(
            {
                SpecialKeys.ID: ids.flatten(),
                "dists": dists.flatten(),
                "train_uid": np.repeat(train_ids, ids.shape[1], axis=0).flatten(),
            }
        )

        self.log_dict(
            {
                "timer/search": elapsed,
                "search/ids_retrieved": len(candidate_df),
                "search/unique_ids_retrieved": candidate_df[SpecialKeys.ID].nunique(),
                "search/num_neighbours": num_neighbours,
            },
            step=self.progress_tracker.global_round,
        )

        return candidate_df

    def get_subpool_ids(
        self, candidate_df: pd.DataFrame, datastore: PandasDataStoreForSequenceClassification
    ) -> List[int]:
        raise NotImplementedError

    def _record_reason(
        self,
        datastore: PandasDataStoreForSequenceClassification,
        candidate_df: pd.DataFrame,
        selected_ids: List[int],
    ) -> None:
        # select those actually to annotate
        annotated_df = candidate_df.loc[candidate_df[SpecialKeys.ID].isin(selected_ids)]
        self._reason_df = pd.concat([annotated_df, self._reason_df], axis=0, ignore_index=False)


class SEALS(BaseSubsetWithSearch):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)  # type: ignore
        self.to_search = []
        self.subpool_ids = []

    def run_query(
        self, model: _FabricModule, datastore: PandasDataStoreForSequenceClassification, query_size: int
    ) -> List[int]:
        selected_ids = super().run_query(model, datastore, query_size)
        self.to_search += selected_ids
        self.subpool_ids = [i for i in self.subpool_ids if i not in selected_ids]
        return selected_ids

    def get_anchors(self, model: _FabricModule, datastore: PandasDataStoreForSequenceClassification) -> List[int]:
        if len(self.to_search) < 1:
            return datastore.get_train_ids()
        return self.to_search

    def get_subpool_ids(
        self, candidate_df: pd.DataFrame, datastore: PandasDataStoreForSequenceClassification
    ) -> List[int]:
        self.subpool_ids += candidate_df[SpecialKeys.ID].unique().tolist()
        return list(set(self.subpool_ids))


class AnchorAL(BaseSubsetWithSearch):
    def __init__(
        self,
        *args,
        anchor_strategy: str,
        only_minority: bool,
        num_anchors: int,
        agg_fn: Optional[str] = None,
        subpool_size: Optional[int] = None,
        subpool_sampling_strategy: Optional[bool] = None,
        temperature: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)  # type: ignore

        self.rng = check_random_state(self.seed)

        # get_train_ids
        self.anchor_strategy = anchor_strategy
        self.only_minority = only_minority
        if anchor_strategy != "all":
            assert num_anchors > 0, f"When anchor_strategy != 'all', num_anchors must be > 0, not {num_anchors}."
        self.num_anchors = num_anchors

        # get_pool_ids
        self.subpool_size = subpool_size
        if subpool_size is not None:
            assert agg_fn is not None, f"When agg_fn is passed, agg_fn must be passed too."
            assert (
                subpool_sampling_strategy is not None
            ), f"When agg_fn is passed, subpool_sampling_strategy must be passed too."
        self.agg_fn = agg_fn
        self.subpool_sampling_strategy = subpool_sampling_strategy
        self.temperature = temperature

    def get_anchors(self, model: _FabricModule, datastore: PandasDataStoreForSequenceClassification) -> List[int]:

        train_ids = datastore.get_train_ids()

        if len(train_ids) == 0 or (self.num_anchors > 0 and len(train_ids) < self.num_anchors):
            self.log("summary/used_anchor_strategy", 0, step=self.progress_tracker.global_round)
            return train_ids

        # decide whether to apply anchor strategy only on the minority
        anchor_ids = train_ids
        if self.only_minority:
            # get minority ids
            train_df = datastore.data.loc[(datastore._train_mask()), [SpecialKeys.ID, InputKeys.TARGET]]
            anchor_ids = train_df.loc[train_df[InputKeys.TARGET] == 1, SpecialKeys.ID].tolist()

        # apply anchor strategy
        if self.anchor_strategy == "all":
            # here just for clarity
            pass

        elif self.anchor_strategy == "random":
            anchor_ids = self.rng.choice(anchor_ids, size=min(self.num_anchors, len(anchor_ids)), replace=False).tolist()  # type: ignore

        elif self.anchor_strategy == "kmeans":
            anchor_ids = run_kmeans(datastore, anchor_ids, num_clusters=self.num_anchors, rng=self.rng, from_pool=False)

        else:
            raise NotImplementedError

        self.log_dict(
            {"summary/num_anchors": len(anchor_ids), "summary/used_anchor_strategy": 1},
            step=self.progress_tracker.global_round,
        )
        return anchor_ids

    def get_subpool_ids(
        self, candidate_df: pd.DataFrame, datastore: PandasDataStoreForSequenceClassification
    ) -> List[int]:

        if self.subpool_size is None:
            return candidate_df[SpecialKeys.ID].unique().tolist()

        if self.subpool_sampling_strategy == "uniform":
            subpool_ids = candidate_df[SpecialKeys.ID].unique().tolist()
            return self.pool_rng.choice(subpool_ids, size=min(self.subpool_size, len(subpool_ids)), replace=False).tolist()  # type: ignore

        # aggregate
        agg_df = (
            candidate_df.groupby(SpecialKeys.ID)  # type: ignore
            .agg(dists=("dists", self.agg_fn), train_uid=("train_uid", "unique"))
            .reset_index()
            # convert cosine distance in similarity (i.e., higher means more similar)
            .assign(scores=lambda _df: 1 - _df["dists"])
        )

        if self.subpool_sampling_strategy == "topk":
            return (
                agg_df.sort_values("scores", ascending=False)
                .head(min(self.subpool_size, len(candidate_df)))[SpecialKeys.ID]
                .tolist()
            )

        elif self.subpool_sampling_strategy == "importance":

            # normalize the scores to probabilities
            probs = softmax(agg_df["scores"] / self.temperature, axis=0)

            # importance sampling
            return self.pool_rng.choice(
                agg_df[SpecialKeys.ID].values,  # type: ignore
                size=min(self.subpool_size, len(agg_df)),
                replace=False,
                p=probs,
            ).tolist()

        raise NotImplementedError
