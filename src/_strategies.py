import math
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from lightning.fabric.wrappers import _FabricModule
from numpy.random import RandomState
from scipy.special import softmax
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state

from energizer.active_learning.datastores.classification import ActivePandasDataStoreForSequenceClassification
from energizer.active_learning.strategies.random import RandomStrategy as _RandomStrategy
from energizer.active_learning.strategies.uncertainty import UncertaintyBasedStrategy as _UncertaintyBasedStrategy
from energizer.enums import InputKeys, SpecialKeys
from src.estimator import SequenceClassificationMixin


def silhouette_k_select(X: np.ndarray, max_k: int, rng: RandomState) -> int:

    silhouette_avg_n_clusters = []
    options = list(range(2, max_k))
    for n_clusters in options:

        # Initialize the clusterer with n_clusters value and a random generator
        clusterer = KMeans(n_clusters=n_clusters, n_init="auto", random_state=rng)
        cluster_labels = clusterer.fit_predict(X)

        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed clusters
        silhouette_avg = silhouette_score(X, cluster_labels)
        silhouette_avg_n_clusters.append(silhouette_avg)

    return options[np.argmax(silhouette_avg_n_clusters).item()]


def run_kmeans(
    X: np.ndarray,
    ids: List[int],
    num_clusters: int,
    rng: RandomState,
    use_silhouette: bool = False,
) -> List[int]:
    """Runs k-means and retrieves the indices of the embeddings closest to the centers."""

    X = StandardScaler().fit_transform(X)

    num_clusters = min(X.shape[0], num_clusters)
    if num_clusters > 1 and use_silhouette:
        num_clusters = silhouette_k_select(X, max_k=num_clusters, rng=rng)

    cluster_learner = KMeans(n_clusters=num_clusters, n_init="auto", random_state=rng)
    cluster_learner.fit(X)
    cluster_idxs = cluster_learner.predict(X)

    # pick instances closest to the cluster centers
    centers = cluster_learner.cluster_centers_[cluster_idxs]
    dists = (X - centers) ** 2
    dists = dists.sum(axis=1)
    closest_ids = [
        np.arange(X.shape[0])[cluster_idxs == i][dists[cluster_idxs == i].argmin()].item() for i in range(num_clusters)
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
        datastore: ActivePandasDataStoreForSequenceClassification,
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
        self, model: _FabricModule, datastore: ActivePandasDataStoreForSequenceClassification, query_size: int
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
        self, model: _FabricModule, datastore: ActivePandasDataStoreForSequenceClassification, query_size: int
    ) -> List[int]:
        subpool_ids = datastore.sample_from_pool(
            size=min(datastore.pool_size(), self.subpool_size), mode="uniform", random_state=self.rng
        )
        num_clusters = query_size * self.r
        embeddings = datastore.get_pool_embeddings(subpool_ids)
        subpool_ids = run_kmeans(embeddings, subpool_ids, num_clusters=num_clusters, rng=self.rng, use_silhouette=False)
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
        self, model: _FabricModule, datastore: ActivePandasDataStoreForSequenceClassification, query_size: int
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

    def get_anchors(self, model: _FabricModule, datastore: ActivePandasDataStoreForSequenceClassification) -> List[int]:
        raise NotImplementedError

    def search_pool(
        self,
        datastore: ActivePandasDataStoreForSequenceClassification,
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
        self, candidate_df: pd.DataFrame, datastore: ActivePandasDataStoreForSequenceClassification
    ) -> List[int]:
        raise NotImplementedError

    def _record_reason(
        self,
        datastore: ActivePandasDataStoreForSequenceClassification,
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
        self, model: _FabricModule, datastore: ActivePandasDataStoreForSequenceClassification, query_size: int
    ) -> List[int]:
        selected_ids = super().run_query(model, datastore, query_size)
        self.to_search += selected_ids
        self.subpool_ids = [i for i in self.subpool_ids if i not in selected_ids]
        return selected_ids

    def get_anchors(self, model: _FabricModule, datastore: ActivePandasDataStoreForSequenceClassification) -> List[int]:
        if len(self.to_search) < 1:
            return datastore.get_train_ids()
        return self.to_search

    def get_subpool_ids(
        self, candidate_df: pd.DataFrame, datastore: ActivePandasDataStoreForSequenceClassification
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

    def get_anchors(self, model: _FabricModule, datastore: ActivePandasDataStoreForSequenceClassification) -> List[int]:

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

        elif self.anchor_strategy in ("kmeans", "kmeans_sil"):
            embeddings = datastore.get_train_embeddings(anchor_ids)
            anchor_ids = run_kmeans(
                embeddings,
                anchor_ids,
                num_clusters=self.num_anchors,
                rng=self.rng,
                use_silhouette=self.anchor_strategy == "kmeans_sil",
            )

        elif self.anchor_strategy in ("diversified", "diversified_rampup"):
            assert self.only_minority is False, "When anchor_strategy == 'diversified', only_minority must be False."

            # get ids
            train_df = datastore.data.loc[(datastore._train_mask()), [SpecialKeys.ID, InputKeys.TARGET]]
            min_ids = train_df.loc[train_df[InputKeys.TARGET] == 1, SpecialKeys.ID].tolist()
            maj_ids = train_df.loc[train_df[InputKeys.TARGET] != 1, SpecialKeys.ID].tolist()

            # select anchors from the minority class
            num_minority_anchors = self.num_anchors
            if self.anchor_strategy == "diversified_rampup" and len(min_ids) >= int(self.num_anchors / 2):
                num_minority_anchors = int(self.num_anchors / 2)
            min_anchors_ids = self.rng.choice(min_ids, size=min(num_minority_anchors, len(min_ids)), replace=False).tolist()  # type: ignore

            # select anchors from the majority class
            num_remaining_anchors = self.num_anchors - num_minority_anchors
            maj_anchors_ids = []
            if num_remaining_anchors > 0:
                maj_instance_loader = self.configure_dataloader(datastore.train_loader(with_indices=maj_ids))  # type: ignore
                self.progress_tracker.pool_tracker.max = len(maj_instance_loader)  # type: ignore
                maj_anchors_ids = self.compute_most_uncertain(model, maj_instance_loader, num_remaining_anchors)  # type: ignore

            anchor_ids = min_anchors_ids + maj_anchors_ids

        else:
            raise NotImplementedError

        self.log_dict(
            {"summary/num_anchors": len(anchor_ids), "summary/used_anchor_strategy": 1},
            step=self.progress_tracker.global_round,
        )
        return anchor_ids

    def get_subpool_ids(
        self, candidate_df: pd.DataFrame, datastore: ActivePandasDataStoreForSequenceClassification
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


class AnchorAL2(BaseSubsetWithSearch):
    def __init__(
        self,
        *args,
        anchor_strategy_minority: str,
        anchor_strategy_majority: str,
        num_anchors: int,
        subpool_size: int,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)  # type: ignore
        self.rng = check_random_state(self.seed)
        self.anchor_strategy_minority = anchor_strategy_minority
        self.anchor_strategy_majority = anchor_strategy_majority
        self.num_anchors = num_anchors
        self.subpool_size = subpool_size

    def run_query(
        self, model: _FabricModule, datastore: ActivePandasDataStoreForSequenceClassification, query_size: int
    ) -> List[int]:

        # GET ANCHORS
        anchor_ids = self.get_anchors(model, datastore)

        if all(len(v) == 0 for v in anchor_ids.values()):
            # if cold-starting there is no training embedding, fall-back to random sampling
            return datastore.sample_from_pool(size=query_size, mode="uniform", random_state=self.pool_rng)

        # SEARCH ANCHORS
        subpool_ids = []
        logs = {}
        candidates_dfs = []
        for k, ids in anchor_ids.items():
            train_embeddings = datastore.get_train_embeddings(ids)
            candidate_df, search_logs = self.search_pool(datastore, train_embeddings, ids)

            # USE RESULTS TO SUBSET POOL
            # NOTE: you are doubling the size of the pool (for each class you are taking self.subpool_size instances)
            subpool_ids += self.get_subpool_ids(candidate_df, datastore)

            logs = {**logs, **{f"{i}_{k}": j for i, j in search_logs.items()}}
            candidates_dfs.append(candidate_df)

        logs["summary/subpool_size"] = len(subpool_ids)

        subpool_ids = list(set(subpool_ids))

        logs["summary/subpool_size_unique"] = len(subpool_ids)

        self.log_dict(logs, step=self.progress_tracker.global_round)

        # RUN ACTIVE LEARNING CRITERION
        selected_ids = self.select(model, datastore, subpool_ids, query_size)

        # add traceability into the datastore
        self._record_reason(datastore, pd.concat(candidates_dfs, axis=0, ignore_index=False), selected_ids)

        return selected_ids

    def get_anchors(
        self, model: _FabricModule, datastore: ActivePandasDataStoreForSequenceClassification
    ) -> Dict[str, List[int]]:

        train_df = datastore.data.loc[(datastore._train_mask()), [SpecialKeys.ID, InputKeys.TARGET]]
        if len(train_df) == 0 or (self.num_anchors > 0 and len(train_df) < self.num_anchors):
            self.log("summary/used_anchor_strategy", 0, step=self.progress_tracker.global_round)
            return {"cold-start": train_df[SpecialKeys.ID].tolist()}

        minority_ids = train_df.loc[train_df[InputKeys.TARGET] == 1, SpecialKeys.ID].tolist() or []
        majority_ids = train_df.loc[train_df[InputKeys.TARGET] != 1, SpecialKeys.ID].tolist() or []

        # NOTE: we always first deal with minority instances
        iterable = {
            "minority": (minority_ids, self.anchor_strategy_minority),
            "majority": (majority_ids, self.anchor_strategy_majority),
        }

        anchor_ids = {}
        num_anchors = self.num_anchors
        for k, (ids, strategy) in iterable.items():

            if num_anchors <= 0:
                continue

            if strategy == "all":
                _ids = ids

            elif strategy == "random":
                _ids = self.rng.choice(ids, size=min(num_anchors, len(ids)), replace=False).tolist()  # type: ignore

            elif strategy in ("kmeans", "kmeans_sil"):
                embeddings = datastore.get_train_embeddings(ids)
                _ids = run_kmeans(
                    embeddings, ids, num_clusters=num_anchors, rng=self.rng, use_silhouette=strategy == "kmeans_sil"
                )

            elif strategy == "uncertainty":
                loader = self.configure_dataloader(datastore.train_loader(with_indices=ids))  # type: ignore
                self.progress_tracker.pool_tracker.max = len(loader)  # type: ignore
                _ids = self.compute_most_uncertain(model, loader, num_anchors)  # type: ignore

            else:
                raise NotImplementedError

            assert len(_ids) == len(set(_ids))  # if we find duplicates, there are bugs
            anchor_ids[k] = _ids
            num_anchors -= len(_ids)

        self.log_dict(
            {**{f"summary/num_{k}_anchors": len(v) for k, v in anchor_ids.items()}, "summary/used_anchor_strategy": 1},
            step=self.progress_tracker.global_round,
        )

        return anchor_ids

    def search_pool(
        self,
        datastore: ActivePandasDataStoreForSequenceClassification,
        query: np.ndarray,
        train_ids: List[int],
    ) -> Tuple[pd.DataFrame, Dict[str, float]]:
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

        logs = {
            "timer/search": elapsed,
            "search/ids_retrieved": len(candidate_df),
            "search/unique_ids_retrieved": candidate_df[SpecialKeys.ID].nunique(),
            "search/num_neighbours": num_neighbours,
        }

        return candidate_df, logs

    def get_subpool_ids(
        self, candidate_df: pd.DataFrame, datastore: ActivePandasDataStoreForSequenceClassification
    ) -> List[int]:

        if self.subpool_size is None:
            return candidate_df[SpecialKeys.ID].unique().tolist()

        return (
            candidate_df.groupby(SpecialKeys.ID)  # type: ignore
            .agg(dists=("dists", "mean"), train_uid=("train_uid", "unique"))
            .reset_index()
            # convert cosine distance in similarity (i.e., higher means more similar)
            .assign(scores=lambda _df: 1 - _df["dists"])
            .sort_values("scores", ascending=False)
            .head(min(self.subpool_size, len(candidate_df)))[SpecialKeys.ID]
            .tolist()
        )
