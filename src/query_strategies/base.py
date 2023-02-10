import pickle
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from lightning.fabric.accelerators.accelerator import Accelerator
from lightning.fabric.connector import _PLUGIN_INPUT, _PRECISION_INPUT
from lightning.fabric.loggers import Logger
from lightning.fabric.strategies import Strategy
from lightning.fabric.wrappers import _FabricDataLoader, _FabricModule
from sklearn.utils.validation import (
    check_random_state,  # https://scikit-learn.org/stable/developers/develop.html#random-numbers
)
from torch.utils.data import DataLoader
from tqdm.auto import tqdm, trange

from src.containers import ActiveFitOutput, BatchOutput, EpochOutput, QueryOutput, RoundOutput
from src.data.active_datamodule import ActiveDataModule
from src.enums import RunningStage, SpecialColumns
from src.estimator import Estimator
from src.registries import SCORING_FUNCTIONS
from src.types import POOL_BATCH_OUTPUT
from src.utilities import get_hparams


class ActiveEstimator(Estimator):

    """
    Active learning loop
    """

    def active_fit(
        self,
        active_datamodule: ActiveDataModule,
        num_rounds: int,
        query_size: int,
        val_perc: float,
        reinit_model: bool = True,
        fit_kwargs: Optional[Dict] = None,
        test_kwargs: Optional[Dict] = None,
        pool_kwargs: Optional[Dict] = None,
        model_cache_dir: Optional[Union[str, Path]] = ".model_cache",
    ) -> ActiveFitOutput:
        # get passed hyper-parameters
        hparams = get_hparams()
        outputs = ActiveFitOutput(hparams=hparams)

        if reinit_model:
            self.save_state_dict(model_cache_dir)

        # hook
        self.fabric.call("on_active_fit_start", output=outputs)

        pbar = self._get_round_progress_bar(num_rounds)
        for round_idx in pbar:
            output = self.round_loop(
                round_idx=round_idx,
                active_datamodule=active_datamodule,
                query_size=query_size,
                fit_kwargs=fit_kwargs,
                test_kwargs=test_kwargs or {},
                pool_kwargs=pool_kwargs or {},
            )
            output.round_idx = round_idx

            outputs.append(output)

            # label data
            if output.query.indices is not None:
                active_datamodule.label(indices=output.query.indices, round_idx=output.round_idx, val_perc=val_perc)

            if reinit_model:
                self.load_state_dict(model_cache_dir)

        # hook
        self.fabric.call("on_active_fit_end", output=outputs)

        return outputs

    def round_loop(
        self,
        active_datamodule: ActiveDataModule,
        query_size: int,
        fit_kwargs: Optional[Dict],
        test_kwargs: Optional[Dict],
        pool_kwargs: Optional[Dict],
    ) -> RoundOutput:
        output = RoundOutput()

        # hook
        self.fabric.call("on_round_start", output=output)

        # fit model on the available data
        if active_datamodule.has_labelled_data:
            output.fit = self.fit(
                train_loader=active_datamodule.train_loader(),
                validation_loader=active_datamodule.validation_loader(),
                **fit_kwargs,
            )

        # test model
        if active_datamodule.has_test_data:
            output.test = self.test(test_loader=active_datamodule.test_loader(), **test_kwargs)

        # query indices to annotate
        if active_datamodule.pool_size > query_size:
            output.query = self.query(active_datamodule=active_datamodule, query_size=query_size, **pool_kwargs)

        # hook
        self.fabric.call("on_round_end", output=output)

        return output

    """
    Query loop
    """

    def query(self, active_datamodule: ActiveDataModule, query_size: int, **kwargs) -> QueryOutput:
        pool_loader = self.get_pool_loader(active_datamodule=active_datamodule)

        model = self.fabric.setup(self.model)
        pool_loader = self.configure_dataloader(pool_loader)

        return self.query_loop(model, pool_loader, query_size, **kwargs)

    def query_loop(
        self,
        model: _FabricModule,
        pool_dataloader: _FabricDataLoader,
        query_size: int,
        **kwargs,
    ) -> QueryOutput:
        raise NotImplementedError

    """
    Methods
    """

    def get_pool_loader(self, active_datamodule: ActiveDataModule) -> DataLoader:
        return active_datamodule.pool_loader()

    def transfer_to_device(self, batch: Any) -> Any:
        # remove string columns that cannot be transfered on gpu
        columns_on_cpu = batch.pop("on_cpu", None)

        # transfer the rest on gpu
        batch = super().transfer_to_device(batch)

        # add the columns on cpu to the batch
        if columns_on_cpu is not None:
            batch["on_cpu"] = columns_on_cpu

        return batch

    def save_state_dict(self, cache_dir: Union[str, Path]) -> None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), cache_dir / "state_dict.pt")

    def load_state_dict(self, cache_dir: Union[str, Path]) -> None:
        cache_dir = Path(cache_dir)
        self.model.load_state_dict(torch.load(cache_dir / "state_dict.pt"))

    """
    Utilities
    """

    def _get_round_progress_bar(self, num_rounds: int) -> tqdm:
        return trange(num_rounds, desc="Completed labelling rounds")

    def _get_epoch_progress_bar(self, num_epochs: int) -> tqdm:
        return trange(num_epochs, desc="Completed epochs", dynamic_ncols=True, leave=False)


class RandomStrategy(ActiveEstimator):
    def __init__(self, seed: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.seed = seed
        self.rng = check_random_state(seed)  # reproducibility

    def query(self, active_datamodule: ActiveDataModule, query_size: int, **kwargs) -> QueryOutput:
        pool_indices = active_datamodule.pool_indices
        indices = self.rng.choice(pool_indices, size=query_size, replace=False)
        return QueryOutput(indices=indices)


class UncertaintyBasedStrategy(ActiveEstimator):
    _scoring_fn_registry = SCORING_FUNCTIONS

    def __init__(self, score_fn: Union[str, Callable], *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if isinstance(score_fn, Callable):
            self.score_fn = score_fn
        else:
            self.score_fn = self._scoring_fn_registry.get(score_fn)

    """
    Query loop
    """

    def query_loop(self, model: _FabricModule, loader: _FabricDataLoader, query_size: int, **kwargs) -> QueryOutput:
        # NOTE: hooks are called within eval_epoch_loop as well as the `pool_step`
        output: EpochOutput = self.eval_epoch_loop(
            model,
            loader,
            stage=RunningStage.POOL,
            dry_run=kwargs.get("dry_run", None),
            limit_batches=kwargs.get("limit_batches", None),
        )
        topk_scores, indices = self._topk(output.output, query_size)

        return QueryOutput(
            topk_scores=topk_scores,
            indices=indices,
            metrics=output.metrics,
            output=output.output,
        )

    def eval_batch_loop(
        self, model: _FabricModule, batch: Any, batch_idx: int, metrics: Any, stage: RunningStage
    ) -> POOL_BATCH_OUTPUT:
        """Hook into the `eval_batch_loop` to automatically add the dataset indices to the output."""

        if stage != RunningStage.POOL:
            return super().eval_batch_loop(model, batch, batch_idx, metrics, stage)

        ids = batch.pop("on_cpu")[SpecialColumns.ID]

        # calls the `pool_step`
        output = super().eval_batch_loop(model, batch, batch_idx, metrics, stage)
        if isinstance(output, torch.Tensor):
            output = {"scores": output}

        output[SpecialColumns.ID] = np.array(ids)

        return output

    """
    Methods
    """

    def pool_step(
        self, model: torch.nn.Module, batch: Any, batch_idx: int, metrics: Optional[Any] = None
    ) -> POOL_BATCH_OUTPUT:
        """Must return a Dict with at least the keys "scores" or "logits"."""
        raise NotImplementedError

    """
    Utilities
    """

    def _topk(self, output: List[BatchOutput], query_size: int) -> Tuple[np.ndarray, List[int]]:
        # get all scores
        all_scores, all_ids = zip(*((i.output["scores"], i.output[SpecialColumns.ID]) for i in output))
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
