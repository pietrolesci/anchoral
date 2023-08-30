from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from lightning.fabric.wrappers import _FabricDataLoader, _FabricModule

from energizer.active_learning.datastores.base import ActiveDataStoreWithIndex, ActivePandasDataStoreWithIndex
from energizer.active_learning.registries import CLUSTERING_FUNCTIONS
from energizer.active_learning.strategies.two_stage import BaseSubsetWithSearchStrategy
from energizer.enums import InputKeys, SpecialKeys


class AnchorAL(BaseSubsetWithSearchStrategy):
    def __init__(
        self,
        *args,
        subpool_size: int,
        seed: int = 42,
        num_neighbours: int,
        max_search_size: Optional[int] = None,
        anchor_strategy_minority: Optional[str] = None,
        anchor_strategy_majority: Optional[str] = None,
        num_anchors: int,
        **kwargs,
    ) -> None:
        super().__init__(
            *args,
            subpool_size=subpool_size,
            seed=seed,
            num_neighbours=num_neighbours,
            max_search_size=max_search_size,
            **kwargs,
        )
        self.anchor_strategy_minority = anchor_strategy_minority
        self.anchor_strategy_majority = anchor_strategy_majority
        self.num_anchors = num_anchors

    def select_search_query(
        self, model: _FabricModule, loader: _FabricDataLoader, datastore: ActivePandasDataStoreWithIndex, **kwargs
    ) -> Dict[str, List[int]]:

        train_df = datastore._train_data.loc[(datastore._train_mask()), [SpecialKeys.ID, InputKeys.TARGET]]
        if len(train_df) == 0 or (self.num_anchors > 0 and len(train_df) < self.num_anchors):
            self.log("summary/used_anchor_strategy", 0, step=self.tracker.global_round)
            return {"cold-start": train_df[SpecialKeys.ID].tolist()}

        minority_ids = train_df.loc[train_df[InputKeys.TARGET] == 1, SpecialKeys.ID].tolist() or []
        majority_ids = train_df.loc[train_df[InputKeys.TARGET] != 1, SpecialKeys.ID].tolist() or []

        iterable = {
            "minority": (minority_ids, self.anchor_strategy_minority),
            "majority": (majority_ids, self.anchor_strategy_majority),
        }

        anchor_ids = {}
        num_anchors = self.num_anchors
        for k, (ids, strategy) in iterable.items():

            if num_anchors <= 0 or strategy is None:
                continue

            if strategy == "all":
                _ids = ids

            elif strategy == "random":
                _ids = self.rng.choice(ids, size=min(num_anchors, len(ids)), replace=False).tolist()  # type: ignore

            # elif strategy == "uncertainty":
            #     loader = self.configure_dataloader(datastore.train_loader(with_indices=ids))  # type: ignore
            #     self.tracker.pool_tracker.max = len(loader)  # type: ignore
            #     _ids = self.compute_most_uncertain(model, loader, num_anchors)  # type: ignore

            elif strategy in CLUSTERING_FUNCTIONS:
                embeddings = datastore.get_train_embeddings(ids)
                cluster_ids = CLUSTERING_FUNCTIONS[strategy](
                    embeddings, num_clusters=min(num_anchors, len(ids)), rng=self.rng
                )
                _ids = [ids[i] for i in cluster_ids]

            else:
                raise NotImplementedError

            assert len(_ids) == len(set(_ids))  # if we find duplicates, there are bugs
            anchor_ids[k] = _ids
            num_anchors -= len(_ids)

        if not len(anchor_ids) > 0:
            anchor_ids = {
                "minority": minority_ids,
                "majority": majority_ids,
            }

        return anchor_ids

    def get_query_embeddings(
        self, datastore: ActiveDataStoreWithIndex, search_query_ids: Dict[str, List[int]]
    ) -> Dict[str, np.ndarray]:
        return {k: datastore.get_train_embeddings(v) for k, v in search_query_ids.items()}

    def search_pool(
        self,
        datastore: ActiveDataStoreWithIndex,
        search_query_embeddings: Dict[str, np.ndarray],
        search_query_ids: Dict[str, List[int]],
    ) -> Dict[str, pd.DataFrame]:
        assert search_query_embeddings.keys() == search_query_ids.keys()
        return {
            k: super().search_pool(datastore, search_query_embeddings[k], search_query_ids[k])
            for k in search_query_embeddings
        }

    def get_subpool_from_search_results(
        self, candidate_df: Dict[str, pd.DataFrame], datastore: ActiveDataStoreWithIndex
    ) -> List[int]:
        def _agg(df: pd.DataFrame) -> pd.DataFrame:
            return (
                df.groupby(SpecialKeys.ID)
                .agg(dists=("dists", "mean"), search_query_uid=("search_query_uid", "unique"))
                .reset_index()
                # convert cosine distance in similarity (i.e., higher means more similar)
                .assign(scores=lambda _df: 1 - _df["dists"])
            )

        def _select(df: pd.DataFrame) -> List[int]:
            return (
                df.sort_values("scores", ascending=False).head(min(self.subpool_size, len(df)))[SpecialKeys.ID].tolist()
            )

        # aggregate by class and then put everything together
        if self.aggregate_by_class:
            list_dfs = [_agg(v_df) for v_df in candidate_df.values()]
            df = pd.concat(list_dfs, ignore_index=False, axis=0)
            # handle duplicates when merging the two datasets (keep only the first because they are sorted)
            df = df.sort_values("scores", ascending=False).drop_duplicates(subset=[SpecialKeys.ID])

        # or put everything together directly
        else:
            df = pd.concat(candidate_df.values(), ignore_index=False, axis=0)
            df = _agg(df)

        return _select(df)
