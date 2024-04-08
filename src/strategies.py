from functools import partial
from typing import Any, Callable, Optional, Union

from lightning.fabric.wrappers import _FabricModule
from torch import Tensor, nn
from torchmetrics import MetricCollection

from energizer.active_learning.strategies.diversity import PoolBasedEmbeddingClustering
from energizer.active_learning.strategies.hybrid import BADGE as _BADGE, Tyrogue as _Tyrogue
from energizer.active_learning.strategies.random import RandomStrategy
from energizer.active_learning.strategies.two_stage import RandomSubsetStrategy, SEALSStrategy
from energizer.active_learning.strategies.uncertainty import UncertaintyBasedStrategy
from energizer.enums import InputKeys
from energizer.types import METRIC
from src.anchoral import (
    AnchorAL,
    AnchorALWithSampling,
    ClusteringAnchorStrategy,
    RandomAnchorStrategy,
    UncertaintyAnchorStrategy,
)
from src.estimator import SequenceClassificationMixin

"""
Instantiate base strategies
"""


class Random(SequenceClassificationMixin, RandomStrategy): ...


class Entropy(SequenceClassificationMixin, UncertaintyBasedStrategy):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, score_fn="entropy", **kwargs)

    def pool_step(
        self,
        model: _FabricModule,
        batch: dict,
        batch_idx: int,
        loss_fn: Optional[Union[nn.Module, Callable]],
        metrics: Optional[MetricCollection] = None,
    ) -> Tensor:
        _ = batch.pop(InputKeys.ON_CPU)  # this is already handled in the `evaluation_step`
        logits = model(**batch).logits
        return self.score_fn(logits)


class EntropyAnchorStrategy(SequenceClassificationMixin, UncertaintyAnchorStrategy):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, score_fn="entropy", **kwargs)

    def pool_step(
        self,
        model: _FabricModule,
        batch: dict,
        batch_idx: int,
        loss_fn: Optional[Union[nn.Module, Callable]],
        metrics: Optional[MetricCollection] = None,
    ) -> Tensor:
        _ = batch.pop(InputKeys.ON_CPU)  # this is already handled in the `evaluation_step`
        logits = model(**batch).logits
        return self.score_fn(logits)


class FTBERTKM(SequenceClassificationMixin, PoolBasedEmbeddingClustering):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, clustering_fn="kmeans_sampling", **kwargs)

    def pool_step(
        self,
        model: _FabricModule,
        batch: Any,
        batch_idx: int,
        loss_fn: Optional[Union[nn.Module, Callable]],
        metrics: Optional[METRIC],
    ) -> Tensor:
        _ = batch.pop(InputKeys.ON_CPU)
        cls_token = model.bert(**batch).last_hidden_state[:, 0, :]
        return cls_token


class BADGE(SequenceClassificationMixin, _BADGE):
    def get_penultimate_layer_out(self, model: _FabricModule, batch: Any) -> Tensor:
        inp = {k: v for k, v in batch.items() if k in (InputKeys.INPUT_IDS, InputKeys.ATT_MASK)}
        return model.bert(**inp).pooler_output

    def get_logits_from_penultimate_layer_out(self, model: _FabricModule, penultimate_layer_out: Tensor) -> Tensor:
        return model.classifier(penultimate_layer_out)


class Tyrogue(SequenceClassificationMixin, _Tyrogue):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(
            *args,
            score_fn="entropy",
            r_factor=3,  # from the paper
            clustering_algorithm="kmeans_sampling",
            clustering_kwargs=None,
            **kwargs,
        )

    def pool_step(
        self,
        model: _FabricModule,
        batch: dict,
        batch_idx: int,
        loss_fn: Optional[Union[nn.Module, Callable]],
        metrics: Optional[MetricCollection] = None,
    ) -> Tensor:
        _ = batch.pop(InputKeys.ON_CPU)  # this is already handled in the `evaluation_step`
        logits = model(**batch).logits
        return self.score_fn(logits)


BASE_STRATEGIES = {"random": Random, "entropy": Entropy, "ftbertkm": FTBERTKM, "badge": BADGE, "tyrogue": Tyrogue}


ANCHOR_STRATEGIES = {
    "random": RandomAnchorStrategy,
    "entropy": EntropyAnchorStrategy,
    "kmeans_sampling": partial(ClusteringAnchorStrategy, clustering_fn="kmeans_sampling"),
    "kmeans_pp_sampling": partial(ClusteringAnchorStrategy, clustering_fn="kmeans_pp_sampling"),
}


"""
Two-stage strategies
"""


class BaseRandomSubset(SequenceClassificationMixin, RandomSubsetStrategy):
    NAME: str = "randomsubset"

    def __init__(self, *args, base_strategy: str, subpool_size: int, seed: int = 42, **kwargs) -> None:
        _base_strategy = BASE_STRATEGIES[base_strategy](*args, seed=seed, **kwargs)
        super().__init__(_base_strategy, subpool_size, seed)


class BaseSEALS(SequenceClassificationMixin, SEALSStrategy):
    NAME: str = "seals"

    def __init__(
        self,
        *args,
        base_strategy: str,
        subpool_size: int,
        seed: int = 42,
        num_neighbours: int,
        max_search_size: Optional[int] = None,
        **kwargs,
    ) -> None:
        _base_strategy = BASE_STRATEGIES[base_strategy](*args, seed=seed, **kwargs)
        super().__init__(
            base_strategy=_base_strategy,
            subpool_size=subpool_size,
            seed=seed,
            num_neighbours=num_neighbours,
            max_search_size=max_search_size,
        )


class BaseAnchorAL(SequenceClassificationMixin, AnchorAL):
    NAME: str = "anchoral"

    def __init__(
        self,
        *args,
        base_strategy: str,
        num_anchors: int,
        anchor_strategy_minority: str,
        anchor_strategy_majority: str,
        minority_classes_ids: Optional[list[int]],
        subpool_size: int,
        seed: int = 42,
        num_neighbours: int,
        max_search_size: Optional[int] = None,
        **kwargs,
    ) -> None:
        _base_strategy = BASE_STRATEGIES[base_strategy](*args, seed=seed, **kwargs)
        _anchor_strategy_minority = ANCHOR_STRATEGIES[anchor_strategy_minority](*args, seed=seed, **kwargs)
        _anchor_strategy_majority = ANCHOR_STRATEGIES[anchor_strategy_majority](*args, seed=seed, **kwargs)
        super().__init__(
            base_strategy=_base_strategy,
            subpool_size=subpool_size,
            seed=seed,
            num_neighbours=num_neighbours,
            max_search_size=max_search_size,
            num_anchors=num_anchors,
            anchor_strategy_minority=_anchor_strategy_minority,
            anchor_strategy_majority=_anchor_strategy_majority,
            minority_classes_ids=minority_classes_ids,
        )


class BaseAnchorALWithSampling(SequenceClassificationMixin, AnchorALWithSampling):
    NAME: str = "anchoralwithsampling"

    def __init__(
        self,
        *args,
        base_strategy: str,
        num_anchors: int,
        anchor_strategy_minority: str,
        anchor_strategy_majority: str,
        minority_classes_ids: Optional[list[int]],
        subpool_size: int,
        seed: int = 42,
        num_neighbours: int,
        max_search_size: Optional[int] = None,
        **kwargs,
    ) -> None:
        _base_strategy = BASE_STRATEGIES[base_strategy](*args, seed=seed, **kwargs)
        _anchor_strategy_minority = ANCHOR_STRATEGIES[anchor_strategy_minority](*args, seed=seed, **kwargs)
        _anchor_strategy_majority = ANCHOR_STRATEGIES[anchor_strategy_majority](*args, seed=seed, **kwargs)
        super().__init__(
            base_strategy=_base_strategy,
            subpool_size=subpool_size,
            seed=seed,
            num_neighbours=num_neighbours,
            max_search_size=max_search_size,
            num_anchors=num_anchors,
            anchor_strategy_minority=_anchor_strategy_minority,
            anchor_strategy_majority=_anchor_strategy_majority,
            minority_classes_ids=minority_classes_ids,
        )


TWOSTAGE_STRATEGIES = {
    f"{cls.NAME}_{base_strategy}": partial(cls, base_strategy=base_strategy)
    for cls in (BaseRandomSubset, BaseSEALS, BaseAnchorAL, BaseAnchorALWithSampling)
    for base_strategy in BASE_STRATEGIES
}

STRATEGIES = {**TWOSTAGE_STRATEGIES, **BASE_STRATEGIES}
