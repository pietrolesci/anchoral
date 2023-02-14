from typing import Any, Callable, List, Optional, Tuple, Union

import numpy as np
import torch
from lightning.fabric.wrappers import _FabricDataLoader, _FabricModule
from sklearn.utils.validation import (
    check_random_state,  # https://scikit-learn.org/stable/developers/develop.html#random-numbers
)
from torch.utils.data import DataLoader

from src.active_learning.base import ActiveEstimator
from src.active_learning.data import ActiveDataModule
from src.containers import EpochOutput, QueryOutput
from src.enums import InputKeys, OutputKeys, RunningStage, SpecialKeys
from src.registries import SCORING_FUNCTIONS
from src.types import EPOCH_OUTPUT, METRIC, POOL_BATCH_OUTPUT


class RandomStrategy(ActiveEstimator):
    def __init__(self, seed: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.seed = seed
        self.rng = check_random_state(seed)  # reproducibility

    def query(self, active_datamodule: ActiveDataModule, query_size: int) -> QueryOutput:
        pool_indices = active_datamodule.pool_indices
        indices = self.rng.choice(pool_indices, size=query_size, replace=False)
        return QueryOutput(indices=indices)


class UncertaintyBasedStrategy(ActiveEstimator):
    _scoring_fn_registry = SCORING_FUNCTIONS

    def __init__(self, score_fn: Union[str, Callable], *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.score_fn = score_fn if isinstance(score_fn, Callable) else self._scoring_fn_registry[score_fn]

    def query_loop(self, model: _FabricModule, loader: _FabricDataLoader, query_size: int, **kwargs) -> QueryOutput:
        """Note that since this relies on the `eval_epoch_loop` method it automatically calls some hooks.

        In particular:
            - configure_metrics
            - on_pool_epoch_start
            - on_pool_batch_start
            - pool_step
            - on_pool_batch_end
            - pool_epoch_end
        """
        output = self.eval_epoch_loop(
            loss_fn=None,
            model=model,
            loader=loader,
            stage=RunningStage.POOL,
            log_interval=kwargs.get("log_interval", 1),
            dry_run=kwargs.get("dry_run", False),
            limit_batches=kwargs.get("limit_pool_batches", None),
        )
        topk_scores, indices = self._topk(output, query_size)

        output = QueryOutput(
            topk_scores=topk_scores,
            indices=indices,
            output=output,
        )

        return output

    def eval_batch_loop(
        self,
        loss_fn: Optional[Union[torch.nn.Module, Callable]],
        model: _FabricModule,
        batch: Any,
        batch_idx: int,
        metrics: Optional[METRIC],
        stage: RunningStage,
    ) -> POOL_BATCH_OUTPUT:
        if stage != RunningStage.POOL:
            return super().eval_batch_loop(loss_fn, model, batch, batch_idx, metrics, stage)

        ids = batch[InputKeys.ON_CPU][SpecialKeys.ID]

        # calls the `pool_step`
        output = super().eval_batch_loop(loss_fn, model, batch, batch_idx, metrics, stage)

        if isinstance(output, torch.Tensor):
            output = {OutputKeys.SCORES: output}
        else:
            assert isinstance(output, dict) and "scores" in output, (
                "In `pool_step` you must return a Tensor with the scores per each element in the batch "
                f"or a Dict with a '{OutputKeys.SCORES}' key and the Tensor of scores as the value."
            )

        output[SpecialKeys.ID] = np.array(ids)

        return output

    """
    Methods
    """

    def pool_step(
        self,
        loss_fn: Optional[Union[torch.nn.Module, Callable]],
        model: _FabricModule,
        batch: Any,
        batch_idx: int,
        metrics: Optional[METRIC] = None,
    ) -> POOL_BATCH_OUTPUT:
        raise NotImplementedError

    def pool_epoch_end(self, output: EPOCH_OUTPUT, metrics: Optional[METRIC]) -> EPOCH_OUTPUT:
        return output

    """
    Utilities
    """

    def _topk(self, output: EPOCH_OUTPUT, query_size: int) -> Tuple[np.ndarray, List[int]]:
        # get all scores
        all_scores, all_ids = zip(*((out["scores"], out[SpecialKeys.ID]) for out in output))
        all_scores = np.concatenate(all_scores)
        all_ids = np.concatenate(all_ids)

        # compute topk
        topk_ids = all_scores.argsort()[-query_size:][::-1]

        topk_scores = all_scores[topk_ids]
        indices = all_ids[topk_ids].tolist()

        return topk_scores, indices


"""
Pool subsampling mixins
"""


class RandomPoolSubsamplingMixin:
    subsampling_size: Union[int, float] = None
    subsampling_rng: int = None

    def get_pool_loader(self, active_datamodule: ActiveDataModule) -> DataLoader:
        pool_indices = active_datamodule.pool_indices
        if isinstance(self.subsampling_size, int):
            pool_size = min(self.subsampling_size, len(pool_indices))
        else:
            pool_size = int(self.subsampling_size * len(pool_indices))

        subset_indices = self.subsampling_rng.choice(pool_indices, size=pool_size)
        return active_datamodule.pool_loader(subset_indices)


class SEALSMixin:
    num_neighbours: int

    def get_pool_loader(self, active_datamodule: ActiveDataModule) -> DataLoader:
        # get the embeddings of the instances not labelled
        train_embeddings = active_datamodule.get_train_embeddings()

        # get neighbours of training instances from the pool
        subset_indices, _ = active_datamodule.index.search_index(
            query=train_embeddings, query_size=self.num_neighbours, query_in_set=False
        )
        subset_indices = np.unique(subset_indices.flatten()).tolist()

        return active_datamodule.pool_loader(subset_indices)


"""
Combined strategies
"""


class RandomSubsamplingRandomStrategy(RandomPoolSubsamplingMixin, RandomStrategy):
    def __init__(self, subsampling_size: Union[int, float], subsampling_seed: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.subsampling_size = subsampling_size
        if isinstance(subsampling_size, float):
            assert 0 < subsampling_size <= 1

        self.subsampling_seed = subsampling_seed
        self.subsampling_rng = check_random_state(subsampling_seed)  # reproducibility


class SEALSRandomStrategy(SEALSMixin, RandomStrategy):
    def __init__(self, num_neighbours: int, *args, **kwargs) -> None:
        super().__init__(self, *args, **kwargs)
        self.num_neighbours = num_neighbours
