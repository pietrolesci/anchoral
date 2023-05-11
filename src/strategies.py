from typing import List

import numpy as np

from energizer.datastores import PandasDataStoreForSequenceClassification
from energizer.strategies import RandomStrategy, UncertaintyBasedStrategy


class RandomStrategySEALS(RandomStrategy):
    def __init__(self, *args, num_neighbours: int = 100, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.num_neighbours = num_neighbours
        self.to_search = []
        self.pool_subset_ids = []

    def run_query(self, *_, datastore: PandasDataStoreForSequenceClassification, query_size: int) -> List[int]:

        if len(self.to_search) == 0:
            self.to_search = datastore.get_train_ids()  # type: ignore

        with_indices = None
        if len(self.to_search) > 0:

            # get the embeddings of the labelled instances
            train_embeddings = datastore.get_embeddings(self.to_search)  # type: ignore

            # get neighbours of training instances from the pool
            nn_ids, _ = datastore.search(  # type: ignore
                query=train_embeddings, query_size=self.num_neighbours, query_in_set=False
            )
            nn_ids = np.unique(np.concatenate(nn_ids).flatten()).tolist()

            with_indices = list(set(nn_ids + self.pool_subset_ids))
            self.pool_subset_ids = with_indices

        annotated_ids = datastore.sample_from_pool(
            size=query_size, mode="uniform", random_state=self.rng, with_indices=with_indices
        )
        self.to_search = annotated_ids  # to search in the next round

        return annotated_ids


class UncertaintyBasedStrategySEALS(UncertaintyBasedStrategy):
    def __init__(self, *args, num_neighbours: int = 100, **kwargs) -> None:
        super().__init__(*args, **kwargs)  # type: ignore
        self.num_neighbours = num_neighbours
        self.to_search = []
        self.pool_subset_ids = []

    def run_query(self, model, datastore: PandasDataStoreForSequenceClassification, query_size: int) -> List[int]:

        if len(self.to_search) == 0:
            self.to_search = datastore.get_train_ids()  # type: ignore

        with_indices = None
        if len(self.to_search) > 0:

            # get the embeddings of the labelled instances
            train_embeddings = datastore.get_embeddings(self.to_search)  # type: ignore

            # get neighbours of training instances from the pool
            nn_ids, _ = datastore.search(  # type: ignore
                query=train_embeddings, query_size=self.num_neighbours, query_in_set=False
            )
            nn_ids = np.unique(np.concatenate(nn_ids).flatten()).tolist()

            with_indices = list(set(nn_ids + self.pool_subset_ids))
            self.pool_subset_ids = with_indices

        pool_loader = self.configure_dataloader(datastore.pool_loader(with_indices=with_indices))
        self.progress_tracker.pool_tracker.max = len(pool_loader)  # type: ignore
        annotated_ids = self.compute_most_uncertain(model, pool_loader, query_size)  # type: ignore
        self.to_search = annotated_ids  # to search in the next round

        return annotated_ids
