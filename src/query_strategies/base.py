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

from src.containers import ActiveFitOutput, BatchOutput, EpochOutput, PoolEpochOutput, RoundOutput
from src.data.active_datamodule import ActiveDataModule
from src.enums import RunningStage, SpecialColumns
from src.estimator import Estimator
from src.registries import SCORING_FUNCTIONS
from src.types import POOL_BATCH_OUTPUT
from src.utilities import Timer, get_hparams


class ActiveEstimator(Estimator):
    def transfer_to_device(self, batch: Any) -> Any:

        # remove string columns that cannot be transfered on gpu
        columns_on_cpu = batch.pop("on_cpu", None)

        # transfer the rest on gpu
        batch = super().transfer_to_device(batch)

        # add the columns on cpu to the batch
        if columns_on_cpu is not None:
            batch["on_cpu"] = columns_on_cpu

        return batch

    def active_fit(
        self,
        active_datamodule: ActiveDataModule,
        num_rounds: int,
        query_size: int,
        val_perc: float,
        fit_kwargs: Optional[Dict] = None,
        test_kwargs: Optional[Dict] = None,
        pool_kwargs: Optional[Dict] = None,
        save_dir: Optional[bool] = None,
    ) -> ActiveFitOutput:

        # get passed hyper-parameters
        hparams = get_hparams()

        fit_kwargs = fit_kwargs or {}
        test_kwargs = test_kwargs or {}
        pool_kwargs = pool_kwargs or {}

        outputs = ActiveFitOutput(hparams=hparams)

        pbar = self._get_round_progress_bar(num_rounds)

        # time the entire active_learning_loop
        with Timer() as t:
            for round_idx in pbar:

                # time the entire round_loop
                with Timer() as round_timer:
                    output = self.round_loop(
                        round_idx=round_idx,
                        active_datamodule=active_datamodule,
                        query_size=query_size,
                        fit_kwargs=fit_kwargs,
                        test_kwargs=test_kwargs,
                        pool_kwargs=pool_kwargs,
                    )
                output.time = round_timer.runtime

                outputs.append(output)

                # maybe save to disk
                if save_dir is not None:
                    self.save_round_output(save_dir, round_idx, output)

                # label data
                if output.pool.indices is not None:
                    active_datamodule.label(indices=output.pool.indices, round_idx=round_idx, val_perc=val_perc)

        outputs.time = t.runtime

        return outputs

    def save_round_output(self, save_dir: str, round: int, output: RoundOutput) -> None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        with (save_dir / f"round_{round}.pkl").open("wb") as fl:
            pickle.dump(output, fl)

    def round_loop(
        self,
        round_idx: int,
        active_datamodule: ActiveDataModule,
        query_size: int,
        fit_kwargs: Optional[Dict],
        test_kwargs: Optional[Dict],
        pool_kwargs: Optional[Dict],
    ) -> RoundOutput:

        output = RoundOutput(round=round_idx)

        if active_datamodule.has_labelled_data:
            output.fit = self.fit(
                train_loader=active_datamodule.train_loader(),
                validation_loader=active_datamodule.validation_loader(),
                **fit_kwargs,
            )

        if active_datamodule.has_test_data:
            output.test = self.test(test_loader=active_datamodule.test_loader(), **test_kwargs)

        if active_datamodule.has_unlabelled_data:

            # query indices to annotate
            pool_output = self.query(
                pool_loader=active_datamodule.pool_loader(),
                query_size=query_size,
                round=round_idx,
                **pool_kwargs,
            )
            output.pool = pool_output

        return output

    def query(
        self,
        pool_loader: DataLoader,
        query_size: int,
        **kwargs,
    ) -> PoolEpochOutput:

        # register dataloader and model with fabric
        model = self.fabric.setup(self.model)
        pool_loader = self.fabric.setup_dataloaders(pool_loader)

        # run pool loop
        with Timer() as t:
            output = self.pool_epoch_loop(model, pool_loader, query_size, **kwargs)

        output.time = t.runtime

        return output

    def pool_epoch_loop(
        self, model: _FabricModule, loader: _FabricDataLoader, query_size: int, **kwargs
    ) -> PoolEpochOutput:
        raise NotImplementedError

    def _get_round_progress_bar(self, num_rounds: int) -> tqdm:
        return trange(num_rounds, desc="Completed labelling rounds")

    def _get_epoch_progress_bar(self, num_epochs: int) -> tqdm:
        return trange(num_epochs, desc="Completed epochs", dynamic_ncols=True, leave=False)


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

    def pool_epoch_loop(
        self, model: _FabricModule, loader: _FabricDataLoader, query_size: int, **kwargs
    ) -> PoolEpochOutput:

        # this will call `eval_batch_loop` that in turn will call `pool_step`
        output: EpochOutput = self.eval_epoch_loop(
            model,
            loader,
            stage=RunningStage.POOL,
            dry_run=kwargs.get("dry_run", None),
            limit_batches=kwargs.get("limit_batches", None),
        )
        topk_scores, indices = self.topk(output.output, query_size)

        output = PoolEpochOutput(
            topk_scores=topk_scores,
            indices=indices,
            metrics=output.metrics,
            output=output.output,
        )

        return output

    def eval_batch_loop(
        self, model: _FabricModule, batch: Any, batch_idx: int, metrics: Any, stage: RunningStage
    ) -> POOL_BATCH_OUTPUT:
        """Hook into the `eval_batch_loop` to automatically add the dataset indices to the output."""
        if stage != RunningStage.POOL:
            return super().eval_batch_loop(model, batch, batch_idx, metrics, stage)

        ids = batch.pop("on_cpu")[SpecialColumns.ID]

        # calls the `pool_step`
        output: POOL_BATCH_OUTPUT = super().eval_batch_loop(model, batch, batch_idx, metrics, stage)
        if "scores" not in output:
            if "logits" in output:
                output["scores"] = self.score_fn(output["logits"])
        else:
            raise KeyError("In `pool_step` you must return a dictionary with either 'scores' or 'logits' key.")

        output[SpecialColumns.ID] = np.array(ids)

        return output

    def topk(self, output: List[BatchOutput], query_size: int) -> Tuple[np.ndarray, List[int]]:
        all_scores, all_ids = zip(*((i.output["scores"], i.output[SpecialColumns.ID]) for i in output))

        all_scores = np.concatenate(all_scores)
        all_ids = np.concatenate(all_ids)

        topk_ids = all_scores.argsort()[-query_size:][::-1]

        topk_scores = all_scores[topk_ids]
        indices = all_ids[topk_ids].tolist()

        return topk_scores, indices

    def pool_step(
        self, model: torch.nn.Module, batch: Any, batch_idx: int, metrics: Optional[Any] = None
    ) -> POOL_BATCH_OUTPUT:
        """Must return a Dict with at least the keys "scores"."""
        raise NotImplementedError
