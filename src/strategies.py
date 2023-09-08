import time
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from lightning.fabric.wrappers import _FabricDataLoader, _FabricModule
from torch import Tensor, nn
from torchmetrics import MetricCollection

from energizer.active_learning.datastores.base import (
    ActiveDataStore,
    ActiveDataStoreWithIndex,
    ActivePandasDataStoreWithIndex,
)
from energizer.active_learning.strategies.diversity import BADGE as _BADGE
from energizer.active_learning.strategies.hybrid import Tyrogue as _Tyrogue
from energizer.active_learning.strategies.random import RandomStrategy
from energizer.active_learning.strategies.two_stage import RandomSubsetStrategy, SEALSStrategy
from energizer.active_learning.strategies.uncertainty import UncertaintyBasedStrategy
from energizer.enums import InputKeys, OutputKeys, SpecialKeys
from src.anchoral import AnchorAL  # , OLDAnchorAL, SimpleAnchorAL
from src.estimator import SequenceClassificationMixin

"""
Instantiate base methods
"""


class Random(SequenceClassificationMixin, RandomStrategy):
    ...


class Entropy(SequenceClassificationMixin, UncertaintyBasedStrategy):
    def __init__(self, *args, seed: int = 42, **kwargs) -> None:
        super().__init__(*args, score_fn="entropy", seed=seed, **kwargs)

    def pool_step(
        self,
        model: _FabricModule,
        batch: Dict,
        batch_idx: int,
        loss_fn: Optional[Union[nn.Module, Callable]],
        metrics: Optional[MetricCollection] = None,
    ) -> Dict:
        _ = batch.pop(InputKeys.ON_CPU)  # this is already handled in the `evaluation_step`
        logits = model(**batch).logits
        scores = self.score_fn(logits)

        return {OutputKeys.SCORES: scores, OutputKeys.LOGITS: logits}


class BADGE(SequenceClassificationMixin, _BADGE):
    def get_penultimate_layer_out(self, model: _FabricModule, batch: Any) -> Tensor:
        inp = {k: v for k, v in batch.items() if k in (InputKeys.INPUT_IDS, InputKeys.ATT_MASK)}
        return model.bert(**inp).pooler_output

    def get_logits_from_penultimate_layer_out(self, model: _FabricModule, penultimate_layer_out: Tensor) -> Tensor:
        return model.classifier(penultimate_layer_out)


class Tyrogue(_Tyrogue):
    def __init__(self, *args, seed: int = 42, **kwargs) -> None:
        super().__init__(
            *args,
            score_fn="entropy",
            seed=seed,
            r_factor=3,  # from the paper
            clustering_algorithm="kmeans_sampling",
            clustering_kwargs=None,
            **kwargs,
        )

    def pool_step(
        self,
        model: _FabricModule,
        batch: Dict,
        batch_idx: int,
        loss_fn: Optional[Union[nn.Module, Callable]],
        metrics: Optional[MetricCollection] = None,
    ) -> Dict:
        _ = batch.pop(InputKeys.ON_CPU)  # this is already handled in the `evaluation_step`
        logits = model(**batch).logits
        scores = self.score_fn(logits)

        return {OutputKeys.SCORES: scores, OutputKeys.LOGITS: logits}


"""
Instantiate two-stage strategies with random subsampling
"""


class LoggingForSubpoolSizeMixin:
    """Wrap `select_pool_subset` to add a call to `self.log_dict`"""

    def select_pool_subset(
        self, model: _FabricModule, loader: _FabricDataLoader, datastore: ActiveDataStore, **kwargs
    ) -> List[int]:
        subpool_ids = super().select_pool_subset(model, loader, datastore, **kwargs)  
        self.log("summary/subpool_size", len(subpool_ids), step=self.tracker.global_round)
        return subpool_ids


class RandomSubsetWithEntropy(SequenceClassificationMixin, LoggingForSubpoolSizeMixin, RandomSubsetStrategy):
    def __init__(self, *args, subpool_size: int, seed: int = 42, **kwargs) -> None:
        base_strategy = Entropy(*args, seed=seed, **kwargs)
        super().__init__(base_strategy, subpool_size, seed)


class RandomSubsetWithBADGE(SequenceClassificationMixin, LoggingForSubpoolSizeMixin, RandomSubsetStrategy):
    def __init__(self, *args, subpool_size: int, seed: int = 42, **kwargs) -> None:
        base_strategy = BADGE(*args, seed=seed, **kwargs)
        super().__init__(base_strategy, subpool_size, seed)


class RandomSubsetWithTyrogue(SequenceClassificationMixin, LoggingForSubpoolSizeMixin, RandomSubsetStrategy):
    def __init__(self, *args, subpool_size: int, seed: int = 42, **kwargs) -> None:
        base_strategy = Tyrogue(*args, seed=seed, **kwargs)
        super().__init__(base_strategy, subpool_size, seed)


"""
Instantiate two-stage strategies with SEALS subsampling
"""


class LoggingForSeachMixin:
    """Wrap `search_pool` to add a call to `self.log_dict`"""

    def search_pool(
        self,
        datastore: ActiveDataStoreWithIndex,
        search_query_embeddings: Dict[str, np.ndarray],
        search_query_ids: Dict[str, List[int]],
    ) -> pd.DataFrame:

        start_time = time.perf_counter()

        search_results = super().search_pool(datastore, search_query_embeddings, search_query_ids)

        ids_retrieved = search_results[SpecialKeys.ID].tolist()
        logs = {
            "timer/search": time.perf_counter() - start_time,
            "search/ids_retrieved": len(ids_retrieved),
            "search/unique_ids_retrieved": len(set(ids_retrieved)),
        }
        self.log_dict(logs, step=self.tracker.global_round)

        return search_results


class SEALSWithEntropy(SequenceClassificationMixin, LoggingForSubpoolSizeMixin, LoggingForSeachMixin, SEALSStrategy):
    def __init__(
        self,
        *args,
        subpool_size: int,
        seed: int = 42,
        num_neighbours: int,
        max_search_size: Optional[int] = None,
        **kwargs,
    ) -> None:
        base_strategy = Entropy(*args, seed=seed, **kwargs)
        super().__init__(
            base_strategy=base_strategy,
            subpool_size=subpool_size,
            seed=seed,
            num_neighbours=num_neighbours,
            max_search_size=max_search_size,
        )


class SEALSWithBADGE(SequenceClassificationMixin, LoggingForSubpoolSizeMixin, LoggingForSeachMixin, SEALSStrategy):
    def __init__(
        self,
        *args,
        subpool_size: int,
        seed: int = 42,
        num_neighbours: int,
        max_search_size: Optional[int] = None,
        **kwargs,
    ) -> None:
        base_strategy = BADGE(*args, seed=seed, **kwargs)
        super().__init__(
            base_strategy=base_strategy,
            subpool_size=subpool_size,
            seed=seed,
            num_neighbours=num_neighbours,
            max_search_size=max_search_size,
        )


class SEALSWithTyrogue(SequenceClassificationMixin, LoggingForSubpoolSizeMixin, SEALSStrategy):
    def __init__(
        self,
        *args,
        subpool_size: int,
        seed: int = 42,
        num_neighbours: int,
        max_search_size: Optional[int] = None,
        **kwargs,
    ) -> None:
        base_strategy = Tyrogue(*args, seed=seed, **kwargs)
        super().__init__(
            base_strategy=base_strategy,
            subpool_size=subpool_size,
            seed=seed,
            num_neighbours=num_neighbours,
            max_search_size=max_search_size,
        )


"""
Instantiate two-stage strategies with AnchorAL subsampling
"""


class LoggingForAnchorsAndSearchMixin:
    """Wrap `select_search_query` to add a call to `self.log_dict`

    Also re-wrap the `search_pool` method because it now returs a dict.
    """

    def select_search_query(
        self, model: _FabricModule, loader: _FabricDataLoader, datastore: ActivePandasDataStoreWithIndex, **kwargs
    ) -> Dict[str, List[int]]:
        search_query = super().select_search_query(model, loader, datastore, **kwargs)  # type: ignore
        self.log_dict(  # type: ignore
            {f"summary/num_{k}_anchors": len(v) for k, v in search_query.items()},
            step=self.tracker.global_round,  # type: ignore
        )
        return search_query

    def search_pool(
        self,
        datastore: ActiveDataStoreWithIndex,
        search_query_embeddings: Dict[str, np.ndarray],
        search_query_ids: Dict[str, List[int]],
    ) -> Dict[str, pd.DataFrame]:

        start_time = time.perf_counter()

        search_results = super().search_pool(datastore, search_query_embeddings, search_query_ids)  # type: ignore
        ids_retrieved = [i for v in search_results.values() for i in v[SpecialKeys.ID].tolist()]

        logs = {
            "timer/search": time.perf_counter() - start_time,
            "search/ids_retrieved": len(ids_retrieved),
            "search/unique_ids_retrieved": len(set(ids_retrieved)),
        }
        self.log_dict(logs, step=self.tracker.global_round)  # type: ignore

        return search_results


class AnchorALWithEntropy(
    SequenceClassificationMixin, LoggingForSubpoolSizeMixin, LoggingForAnchorsAndSearchMixin, AnchorAL
):
    def __init__(
        self,
        *args,
        num_anchors: int,
        anchor_strategy: str,
        anchor_strategy_minority: str,
        minority_classes_ids: Optional[List[int]],
        subpool_size: int,
        seed: int = 42,
        num_neighbours: int,
        max_search_size: Optional[int] = None,
        **kwargs,
    ) -> None:
        base_strategy = Entropy(*args, seed=seed, **kwargs)
        super().__init__(
            base_strategy=base_strategy,
            subpool_size=subpool_size,
            seed=seed,
            num_neighbours=num_neighbours,
            max_search_size=max_search_size,
            num_anchors=num_anchors,
            anchor_strategy=anchor_strategy,
            anchor_strategy_minority=anchor_strategy_minority,
            minority_classes_ids=minority_classes_ids,
        )


class AnchorALWithBADGE(
    SequenceClassificationMixin, LoggingForSubpoolSizeMixin, LoggingForAnchorsAndSearchMixin, AnchorAL
):
    def __init__(
        self,
        *args,
        num_anchors: int,
        anchor_strategy: str,
        anchor_strategy_minority: str,
        minority_classes_ids: Optional[List[int]],
        subpool_size: int,
        seed: int = 42,
        num_neighbours: int,
        max_search_size: Optional[int] = None,
        **kwargs,
    ) -> None:
        base_strategy = BADGE(*args, seed=seed, **kwargs)
        super().__init__(
            base_strategy=base_strategy,
            subpool_size=subpool_size,
            seed=seed,
            num_neighbours=num_neighbours,
            max_search_size=max_search_size,
            num_anchors=num_anchors,
            anchor_strategy=anchor_strategy,
            anchor_strategy_minority=anchor_strategy_minority,
            minority_classes_ids=minority_classes_ids,
        )


class AnchorALWithTyrogue(
    SequenceClassificationMixin, LoggingForSubpoolSizeMixin, LoggingForAnchorsAndSearchMixin, AnchorAL
):
    def __init__(
        self,
        *args,
        num_anchors: int,
        anchor_strategy: str,
        anchor_strategy_minority: str,
        minority_classes_ids: Optional[List[int]],
        subpool_size: int,
        seed: int = 42,
        num_neighbours: int,
        max_search_size: Optional[int] = None,
        **kwargs,
    ) -> None:
        base_strategy = Tyrogue(*args, seed=seed, **kwargs)
        super().__init__(
            base_strategy=base_strategy,
            subpool_size=subpool_size,
            seed=seed,
            num_neighbours=num_neighbours,
            max_search_size=max_search_size,
            num_anchors=num_anchors,
            anchor_strategy=anchor_strategy,
            anchor_strategy_minority=anchor_strategy_minority,
            minority_classes_ids=minority_classes_ids,
        )


# class OLDAnchorALWithEntropy(
#     SequenceClassificationMixin,
#     LoggingForSubpoolSizeMixin,
#     LoggingForAnchorsAndSearchMixin,
#     OLDAnchorAL,
# ):
#     def __init__(
#         self,
#         *args,
#         subpool_size: int,
#         seed: int = 42,
#         num_neighbours: int,
#         max_search_size: Optional[int] = None,
#         anchor_strategy_minority: Optional[str] = None,
#         anchor_strategy_majority: Optional[str] = None,
#         num_anchors: int,
#         aggregate_by_class_first: bool,
#         **kwargs,
#     ) -> None:
#         base_strategy = Entropy(*args, seed=seed, **kwargs)
#         super().__init__(
#             base_strategy=base_strategy,
#             subpool_size=subpool_size,
#             seed=seed,
#             num_neighbours=num_neighbours,
#             max_search_size=max_search_size,
#             anchor_strategy_minority=anchor_strategy_minority,
#             anchor_strategy_majority=anchor_strategy_majority,
#             num_anchors=num_anchors,
#             aggregate_by_class_first=aggregate_by_class_first,
#         )


# class OLDAnchorALWithBADGE(
#     SequenceClassificationMixin,
#     LoggingForSubpoolSizeMixin,
#     LoggingForAnchorsAndSearchMixin,
#     OLDAnchorAL,
# ):
#     def __init__(
#         self,
#         *args,
#         subpool_size: int,
#         seed: int = 42,
#         num_neighbours: int,
#         max_search_size: Optional[int] = None,
#         anchor_strategy_minority: Optional[str] = None,
#         anchor_strategy_majority: Optional[str] = None,
#         num_anchors: int,
#         aggregate_by_class_first: bool,
#         **kwargs,
#     ) -> None:
#         base_strategy = BADGE(*args, seed=seed, **kwargs)
#         super().__init__(
#             base_strategy=base_strategy,
#             subpool_size=subpool_size,
#             seed=seed,
#             num_neighbours=num_neighbours,
#             max_search_size=max_search_size,
#             anchor_strategy_minority=anchor_strategy_minority,
#             anchor_strategy_majority=anchor_strategy_majority,
#             num_anchors=num_anchors,
#             aggregate_by_class_first=aggregate_by_class_first,
#         )


# class SimpleAnchorALWithEntropy(SequenceClassificationMixin, LoggingForSubpoolSizeMixin, SimpleAnchorAL):
#     def __init__(
#         self,
#         *args,
#         subpool_size: int,
#         seed: int = 42,
#         num_anchors: int,
#         anchor_strategy: str,
#         num_neighbours: int,
#         max_search_size: Optional[int] = None,
#         **kwargs,
#     ) -> None:
#         base_strategy = Entropy(*args, seed=seed, **kwargs)
#         super().__init__(
#             base_strategy=base_strategy,
#             subpool_size=subpool_size,
#             seed=seed,
#             num_neighbours=num_neighbours,
#             max_search_size=max_search_size,
#             num_anchors=num_anchors,
#             anchor_strategy=anchor_strategy,
#         )
