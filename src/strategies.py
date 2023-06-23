import math
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from lightning.fabric.wrappers import _FabricModule
from scipy.special import softmax
from sklearn.utils import check_random_state

from energizer.datastores import PandasDataStoreForSequenceClassification
from energizer.enums import InputKeys, SpecialKeys
from energizer.strategies import RandomStrategy as _RandomStrategy
from energizer.strategies import UncertaintyBasedStrategy as _UncertaintyBasedStrategy
from src.estimators import SequenceClassificationMixin
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize


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
        pool_loader = self.configure_dataloader(datastore.pool_loader(with_indices=subset_ids))  # type: ignore
        self.progress_tracker.pool_tracker.max = len(pool_loader)  # type: ignore
        return self.compute_most_uncertain(model, pool_loader, query_size)  # type: ignore


class RandomSubset(SequenceClassificationMixin, UncertaintyMixin, _UncertaintyBasedStrategy):
    def __init__(self, *args, subset_size: int, seed: int, **kwargs) -> None:
        """Strategy that runs uncertainty sampling on a random subset of the pool.

        Args:
            subset_size (int): Size of the subset.
            seed (int): Random seed for the subset selection.
        """
        super().__init__(*args, **kwargs)
        self.subset_size = subset_size
        self.seed = seed
        self.rng = check_random_state(seed)

    def run_query(
        self, model: _FabricModule, datastore: PandasDataStoreForSequenceClassification, query_size: int
    ) -> List[int]:
        subset_size = min(datastore.pool_size(), self.subset_size)
        subset_ids = datastore.sample_from_pool(size=subset_size, mode="uniform", random_state=self.rng)
        self.log("summary/subset_size", len(subset_ids), step=self.progress_tracker.global_round)
        return self.select(model, datastore, subset_ids, query_size)


class AnchorsSubset(SequenceClassificationMixin, UncertaintyMixin, _UncertaintyBasedStrategy):
    _reason_df: pd.DataFrame = pd.DataFrame()

    def __init__(
        self,
        *args,
        subset_size: int,
        anchor_strategy: str,
        num_anchors: Optional[int],
        num_neighbours: int,
        max_search_size: int,
        agg_fn: str,
        seed: int,
        pad_subset: bool,
        **kwargs,
    ) -> None:
        """Strategy that uses anchor points from the training to restrict the pool and run uncertainty sampling.

        Args:
            subset_size (int): Size of the subset.
            anchor_strategy (str): How to select the anchor points. Possible values are `latest`, `random`, `all`.
            num_anchors (Optional[int]): How many anchor points to use as a query when `anchor_strategy` is `random`. When
                the `anchor_strategy` is `latest` this will implicitly be the `query_size`. When `full` it is equal
                to the training size that grows over time.
            num_neighbours (int): How many points to retrieve for each anchor.
            max_search_size (int): Limits the total number of neighbours to retrieve to help with speed.
            agg_fn (str): How to aggregate the distances when the same pool point is retrieved by different training
                points. Possible values are those accepted by Pandas groupby function (max, min, mean, median, etc).
            seed (int): Random seed.
        """
        super().__init__(*args, **kwargs)

        self.seed = seed
        self.rng = check_random_state(seed)
        self.pool_rng = check_random_state(seed)

        # get_train_ids
        if anchor_strategy == "random":
            assert num_anchors is not None, ValueError("When anchor_strategy == random, num_anchors must be set.")
        self.anchor_strategy = anchor_strategy
        self.num_anchors = num_anchors

        # search
        self.num_neighbours = num_neighbours
        self.max_search_size = max_search_size

        # get_pool_ids
        self.subset_size = subset_size
        self.agg_fn = agg_fn
        self.pad_subset = pad_subset

        self.to_search: List[int] = []

    def run_query(
        self, model: _FabricModule, datastore: PandasDataStoreForSequenceClassification, query_size: int
    ) -> List[int]:

        # GET ANCHORS
        train_ids = self.get_train_ids(model, datastore)
        self.log("summary/num_train_ids", len(train_ids), step=self.progress_tracker.global_round)

        if len(train_ids) == 0:
            # if cold-starting there is no training embedding, fall-back to random sampling
            return datastore.sample_from_pool(size=query_size, mode="uniform", random_state=self.rng)

        # SEARCH ANCHORS
        train_embeddings = datastore.get_train_embeddings(train_ids)
        ids, dists = self.search_pool(datastore, train_embeddings)
        anchors_df = self._get_dataframe(ids, dists, train_ids)

        # USE RESULTS TO SUBSET POOL
        subset_ids = self.get_pool_ids(anchors_df, datastore)

        # logs
        self.log_dict(
            {
                "summary/ids_retrieved": len(anchors_df),
                "summary/unique_ids_retrieved": anchors_df[SpecialKeys.ID].nunique(),
                "summary/subset_size": len(subset_ids),
            },
            step=self.progress_tracker.global_round,
        )

        # SELECT DATA TO ANNOTATE
        selected_ids = self.select(model, datastore, subset_ids, query_size)

        # record for the next round in case the anchor_strategy is "latest"
        self.to_search = selected_ids
        # add traceability into the datastore
        self._record_reason(datastore, anchors_df, selected_ids)

        return selected_ids

    def get_train_ids(self, model: _FabricModule, datastore: PandasDataStoreForSequenceClassification) -> List[int]:

        train_ids = datastore.get_train_ids()

        if (
            len(train_ids) == 0
            or self.anchor_strategy == "all"
            or (self.anchor_strategy == "latest" and len(self.to_search) == 0)
            or (self.num_anchors is not None and len(train_ids) < self.num_anchors)
        ):
            return train_ids

        elif self.anchor_strategy == "latest":
            return self.to_search

        elif self.anchor_strategy == "random":
            return self.rng.choice(train_ids, size=min(self.num_anchors, len(train_ids)), replace=False).tolist()  # type: ignore

        elif self.anchor_strategy == "uncertain":
            train_loader = self.configure_dataloader(datastore.train_loader())  # type: ignore
            self.progress_tracker.pool_tracker.max = len(train_loader)  # type: ignore
            return self.compute_most_uncertain(model, train_loader, self.num_anchors)  # type: ignore

        elif self.anchor_strategy == "all_positive":
            train_df = datastore.data.loc[(datastore._train_mask()), [SpecialKeys.ID, InputKeys.TARGET]]
            ids = train_df.loc[train_df[InputKeys.TARGET] == 1, SpecialKeys.ID].tolist()
            return ids if self.num_anchors is None else self.rng.choice(ids, size=min(self.num_anchors, len(ids)), replace=False)  # type: ignore

        elif self.anchor_strategy == "all_positive_uncertain":
            pos_train_ids = datastore.data.loc[
                (datastore._train_mask()) & (datastore.data[InputKeys.TARGET] == 1), SpecialKeys.ID
            ].tolist()

            train_loader = self.configure_dataloader(datastore.train_loader(with_indices=pos_train_ids))  # type: ignore
            self.progress_tracker.pool_tracker.max = len(train_loader)  # type: ignore
            return self.compute_most_uncertain(model, train_loader, self.num_anchors)  # type: ignore

        elif self.anchor_strategy == "kmeans":
            embeddings = datastore.get_train_embeddings(train_ids)
            embeddings: np.ndarray = normalize(embeddings, axis=1)  # type: ignore
            num_clusters = min(embeddings.shape[0], self.num_anchors)  # type: ignore

            cluster_learner = KMeans(n_clusters=num_clusters, n_init="auto", random_state=self.rng)
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
            return np.array(train_ids)[closest_ids].tolist()

        raise NotImplementedError

    def search_pool(
        self, datastore: PandasDataStoreForSequenceClassification, query: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get neighbours of training instances from the pool."""
        # FIXME: Add timer for search
        num_neighbours = min(self.num_neighbours, math.floor(self.max_search_size / query.shape[0]))  # FIXME: this can create noise in the experimentss
        return datastore.search(query=query, query_size=num_neighbours, query_in_set=False)

    def get_pool_ids(self, anchors_df: pd.DataFrame, datastore: PandasDataStoreForSequenceClassification) -> List[int]:
        """Given all the matches get the `subset_size` closest."""

        # aggregate
        df = (
            anchors_df.groupby(SpecialKeys.ID)
            .agg(dists=("dists", self.agg_fn), train_uid=("train_uid", "unique"))
            .reset_index()
        )

        # take top-k and return uid
        subset_ids = df.sort_values("dists", ascending=False).head(self.subset_size)[SpecialKeys.ID].tolist()

        if self.pad_subset:
            subset_ids += self._pad_subset(datastore.get_pool_ids(), subset_ids)

        return subset_ids

    def _pad_subset(self, pool_ids: List[int], subset_ids: List[int]) -> List[int]:
        num_samples = self.subset_size - len(subset_ids)
        ids = list(set(pool_ids).difference(subset_ids))
        return self.pool_rng.choice(ids, size=min(num_samples, len(ids)), replace=False).tolist()

    def _get_dataframe(self, ids: np.ndarray, distances: np.ndarray, train_ids: List[int]) -> pd.DataFrame:
        return pd.DataFrame(
            {
                SpecialKeys.ID: ids.flatten(),
                "dists": distances.flatten(),
                "train_uid": np.repeat(train_ids, ids.shape[1], axis=0).flatten(),
            }
        )

    def _record_reason(
        self,
        datastore: PandasDataStoreForSequenceClassification,
        anchors_df: pd.DataFrame,
        selected_ids: List[int],
    ) -> None:

        # select those actually to annotate
        annotated_df = anchors_df.loc[anchors_df[SpecialKeys.ID].isin(selected_ids)]
        self._reason_df = pd.concat([annotated_df, self._reason_df], axis=0, ignore_index=False)


class AnchorsSubsetWithSampling(AnchorsSubset):
    def __init__(self, *args, temperature: float, **kwargs) -> None:
        """Same as AnchorsSubset but adds sampling instead of top-k.

        It samples globally, that is not class-conditioned.

        Args:
            temperature (float): Temperature parameter.
        """
        super().__init__(*args, **kwargs)
        self.temperature = temperature
        self.pool_rng = check_random_state(self.seed)

    def get_pool_ids(self, anchors_df: pd.DataFrame, datastore: PandasDataStoreForSequenceClassification) -> List[int]:
        # aggregate
        df = (
            anchors_df.groupby(SpecialKeys.ID)
            .agg(dists=("dists", self.agg_fn), train_uid=("train_uid", "unique"))
            .reset_index()
        )

        # convert cosine distance in similarity (i.e., higher means more similar)
        scores = 1 - df["dists"]

        # normalize the scores to probabilities
        probs = softmax(scores / self.temperature, axis=0)

        subset_ids = self.pool_rng.choice(df[SpecialKeys.ID].values, size=min(self.subset_size, len(df)), replace=False, p=probs).tolist()  # type: ignore

        if self.pad_subset:
            random_samples = self._pad_subset(datastore.get_pool_ids(), subset_ids)
            self.log("summary/unique_random_samples", len(set(random_samples)), step=self.progress_tracker.global_round)
            subset_ids += random_samples

        return subset_ids


class AnchorsSubsetWithPerClassSampling(AnchorsSubset):
    def __init__(
        self, *args, positive_class_subset_prop: float, negative_strategy: bool, temperatures: List[float], **kwargs
    ) -> None:
        """Same as AnchorsSubset but adds sampling instead of top-k.

        It samples in a class-conditional way.

        Args:
            positive_class_subset_prop (float): Proportion of the `subset_size` that needs to be allocated
                to the positive class.
            negative_strategy (bool): Whether to sample the negative class assigning higher probability
                to dissimilar instances (higher = most dissimilar).
            temperatures (List[float]): Temperature parameter for each class.
        """
        super().__init__(*args, **kwargs)
        self.positive_class_subset_prop = positive_class_subset_prop
        self.temperatures = temperatures
        self.negative_strategy = negative_strategy
        self.pool_rng = check_random_state(self.seed)

    def get_pool_ids(self, anchors_df: pd.DataFrame, datastore: PandasDataStoreForSequenceClassification) -> List[int]:

        # bring in the class of the training instances that caused the pool instance to be chosen
        # FIXME: PAY ATTENTION -- when you use random and kmeans anchor strategies, this part of the implementation
        # still filters for the candidates that were queries by minority instances. So if you do not fix it, both
        # these strategies can pick up less minority instances and thus, effectively, use less anchor points!
        # THIS LOGIC should be moved in get_train_ids because it is effectively affecting which anchors are used
        df = pd.merge(
            anchors_df,
            datastore.data.loc[datastore._train_mask(), [SpecialKeys.ID, InputKeys.TARGET]],
            left_on="train_uid",
            right_on="uid",
            suffixes=["", "_drop"],
            how="left",
        ).drop(columns=[f"{SpecialKeys.ID}_drop"])

        # aggregate per class
        df = (
            df.groupby([SpecialKeys.ID, InputKeys.TARGET])
            .agg(dists=("dists", self.agg_fn), train_uid=("train_uid", "unique"))
            .reset_index()
        )

        samples = []
        logs = {
            "summary/unique_pos_samples": 0,
            "summary/unique_neg_samples": 0,
            "summary/unique_random_samples": 0,
        }

        # sample positive class
        pos_df = df.loc[df[InputKeys.TARGET] == 1]
        if len(pos_df) > 0:
            num_samples = math.ceil(self.positive_class_subset_prop * self.subset_size)
            pos_samples = self._sample(pos_df, self.temperatures[1], num_samples, False)
            logs["summary/unique_pos_samples"] = len(pos_samples)
            samples += pos_samples

        # sample negative class
        neg_df = df.loc[(df[InputKeys.TARGET] != 1) & (~df[SpecialKeys.ID].isin(samples))]
        if len(neg_df) > 0 and self.negative_strategy is not None:
            dissimilar = self.negative_strategy == "dissimilar"
            num_samples = min(self.subset_size - len(samples), len(neg_df))
            neg_samples = self._sample(neg_df, self.temperatures[0], num_samples, dissimilar)
            logs["summary/unique_neg_samples"] = len(neg_samples)
            samples += neg_samples
            samples = list(set(samples))

        samples = list(set(samples))
        if self.pad_subset or len(samples) == 0:
            random_samples = self._pad_subset(datastore.get_pool_ids(), samples)
            logs["summary/unique_random_samples"] = len(random_samples)
            samples += random_samples
            samples = list(set(samples))

        self.log_dict(logs, step=self.progress_tracker.global_round)

        return samples

    def _sample(
        self, df: pd.DataFrame, temperature: float, num_samples: int, dissimilar: Optional[bool] = False
    ) -> List[int]:

        # convert cosine distance in similarity (i.e., higher means more similar)
        scores = df["dists"] if dissimilar else 1 - df["dists"]

        # normalize the scores to probabilities
        probs = softmax(scores / temperature, axis=0)

        num_samples = min(num_samples, len(df))
        samples = self.pool_rng.choice(df[SpecialKeys.ID], size=num_samples, replace=False, p=probs).tolist()

        return list(set(samples))
