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
from torch.utils.data import DataLoader
from tqdm.auto import tqdm, trange

from src.containers import ActiveFitOutput, BatchOutput, EpochOutput, QueryOutput, RoundOutput
from src.data.active_datamodule import ActiveDataModule
from src.enums import RunningStage, SpecialColumns
from src.estimator import Estimator
from src.registries import SCORING_FUNCTIONS
from src.types import POOL_BATCH_OUTPUT
from src.utilities import Timer, get_hparams


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
            output.query = self.query(
                active_datamodule=active_datamodule,
                query_size=query_size,
                **pool_kwargs,
            )

        # hook
        self.fabric.call("on_round_end", output=output)

        return output

    """
    Query loop
    """

    def query(self, actve_datamodule: ActiveDataModule, query_size: int, **kwargs) -> QueryOutput:
        raise NotImplementedError

    """
    Methods
    """

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
    def query(self, actve_datamodule: ActiveDataModule, query_size: int, **kwargs) -> QueryOutput:
        pool_indices = actve_datamodule.pool_indices
        indices = np.random.choice(pool_indices, size=query_size, replace=False)
        return QueryOutput(indices=indices)


class UncertaintyBasedStrategy(ActiveEstimator):
    _scoring_fn_registry = SCORING_FUNCTIONS

    def __init__(
        self,
        model: torch.nn.Module,
        score_fn: Union[str, Callable],
        accelerator: Optional[Union[str, Accelerator]] = None,
        strategy: Optional[Union[str, Strategy]] = None,
        devices: Optional[Union[List[int], str, int]] = None,
        num_nodes: int = 1,
        precision: _PRECISION_INPUT = 32,
        plugins: Optional[Union[_PLUGIN_INPUT, List[_PLUGIN_INPUT]]] = None,
        callbacks: Optional[Union[List[Any], Any]] = None,
        loggers: Optional[Union[Logger, List[Logger]]] = None,
        deterministic: bool = True,
    ) -> None:
        super().__init__(
            model, accelerator, strategy, devices, num_nodes, precision, plugins, callbacks, loggers, deterministic
        )
        if isinstance(score_fn, Callable):
            self.score_fn = score_fn
        else:
            self.score_fn = self._scoring_fn_registry.get(score_fn)

    """
    Query loop
    """

    def query(self, actve_datamodule: ActiveDataModule, query_size: int, **kwargs) -> QueryOutput:
        # register dataloader and model with fabric
        model = self.fabric.setup(self.model)
        pool_loader = self.configure_dataloader(actve_datamodule.pool_loader())

        return self.pool_epoch_loop(model, pool_loader, query_size, **kwargs)

    def pool_epoch_loop(
        self, model: _FabricModule, loader: _FabricDataLoader, query_size: int, **kwargs
    ) -> QueryOutput:

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
