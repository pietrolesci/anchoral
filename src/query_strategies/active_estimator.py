import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import trange

from src.containers import EpochOutput, EvaluationOutput, PoolEpochOutput
from src.data.active_datamodule import ActiveDataModule
from src.enums import RunningStage, SpecialColumns
from src.estimators.estimator import Estimator
from src.types import POOL_BATCH_OUTPUT
from src.utilities import get_hparams


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
        fit_kwargs: Dict,
        test_kwargs: Dict,
        pool_kwargs: Dict,
        save_dir: Optional[bool] = None,
    ):

        for round_idx in trange(num_rounds, desc="Completed labelling rounds"):

            fit_output = None
            if active_datamodule.has_labelled_data:
                fit_output = self.fit(
                    train_loader=active_datamodule.train_dataloader(),
                    validation_loader=active_datamodule.val_dataloader(),
                    **fit_kwargs,
                )

            test_output = None
            if active_datamodule.has_test_data:
                test_output = self.test(test_loader=active_datamodule.test_dataloader(), **test_kwargs)

            pool_output = None
            if active_datamodule.has_unlabelled_data:

                # query indices to annotate
                pool_output = self.query(
                    pool_loader=active_datamodule.pool_dataloader(), query_size=query_size, round=round_idx, **pool_kwargs
                )
                indices = pool_output.indices

                # label data
                active_datamodule.label(indices=indices, round_id=round_idx, val_perc=val_perc)

            if save_dir is not None:
                data = {
                    "round": round_idx,
                    "fit_output": fit_output,
                    "test_output": test_output,
                    "pool_output": pool_output,
                }
                with (Path(save_dir) / f"outputs_round_{round_idx}").open("wb") as fl:
                    pickle.dump(data, fl)

    def query(
        self,
        pool_loader: DataLoader,
        query_size: int,
        round: int,
        dry_run: Optional[bool] = None,
        limit_batches: Optional[int] = None,
    ) -> PoolEpochOutput:
        raise NotImplementedError


class PoolBasedActiveEstimator(ActiveEstimator):
    def query(
        self,
        pool_loader: DataLoader,
        query_size: int,
        round: int,
        dry_run: Optional[bool] = None,
        limit_batches: Optional[int] = None,
    ) -> PoolEpochOutput:

        # get passed hyper-parameters
        hparams = get_hparams()

        # register dataloader and model with fabric
        model = self.fabric.setup(self.model)
        pool_loader = self.fabric.setup_dataloaders(pool_loader)

        # run pool
        output = PoolEpochOutput(round=round)
        eval_output: EpochOutput = self.eval_epoch_loop(
            model, pool_loader, stage=RunningStage.POOL, dry_run=dry_run, limit_batches=limit_batches
        )
        output.from_epoch_output(eval_output)

        # compute scores
        output = self.aggregate(output, query_size)

        return output

    def eval_batch_loop(
        self, model: torch.nn.Module, batch: Any, batch_idx: int, metrics: Any, stage: RunningStage
    ) -> POOL_BATCH_OUTPUT:
        if stage != RunningStage.POOL:
            return super().eval_batch_loop(model, batch, batch_idx, metrics, stage)

        ids = batch.pop("on_cpu")[SpecialColumns.ID]

        output: POOL_BATCH_OUTPUT = super().eval_batch_loop(model, batch, batch_idx, metrics, stage)
        assert "scores" in output, KeyError("In `pool_step` you must return a dictionary with the 'scores' key.")
        
        output[SpecialColumns.ID] = np.array(ids)

        return output

    def pool_step(
        self, model: torch.nn.Module, batch: Any, batch_idx: int, metrics: Optional[Any] = None
    ) -> POOL_BATCH_OUTPUT:
        raise NotImplementedError

    def aggregate(self, output: PoolEpochOutput, query_size: int) -> PoolEpochOutput:
        all_scores = np.concatenate([i["scores"] for i in output.output])
        all_ids = np.concatenate([i[SpecialColumns.ID] for i in output.output])

        topk_ids = all_scores.argsort()[-query_size:][::-1]

        output.topk_scores = all_scores[topk_ids]
        output.indices = all_ids[topk_ids].tolist()

        return output
