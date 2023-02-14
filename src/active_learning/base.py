from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union

import torch
from lightning.fabric.wrappers import _FabricDataLoader, _FabricModule
from torch.utils.data import DataLoader
from tqdm.auto import tqdm, trange

from src.active_learning.data import ActiveDataModule
from src.containers import ActiveCounter, ActiveFitOutput, QueryOutput, RoundOutput
from src.estimator import Estimator
from src.utilities import get_hparams


class ActiveEstimator(Estimator):
    _counter: ActiveCounter = None

    @property
    def counter(self) -> ActiveCounter:
        if self._counter is None:
            self._counter = ActiveCounter()
        return self._counter

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
        model_cache_dir: Optional[Union[str, Path]] = ".model_cache",
        num_epochs: Optional[int] = 3,
        loss_fn: Optional[Union[str, torch.nn.Module, Callable]] = None,
        loss_fn_kwargs: Optional[Dict] = None,
        learning_rate: float = 0.001,
        optimizer: str = "adamw",
        optimizer_kwargs: Optional[Dict] = None,
        scheduler: Optional[str] = None,
        scheduler_kwargs: Optional[Dict] = None,
        log_interval: int = 1,
        limit_train_batches: Optional[int] = None,
        limit_validation_batches: Optional[int] = None,
        limit_test_batches: Optional[int] = None,
        dry_run: Optional[bool] = False,
    ) -> ActiveFitOutput:
        # get passed hyper-parameters
        hparams = get_hparams()
        outputs = ActiveFitOutput(hparams=hparams)

        if reinit_model:
            self.save_state_dict(model_cache_dir)

        # reset counter
        self.counter.reset()

        pbar = self._get_round_progress_bar(num_rounds)

        # hook
        self.fabric.call("on_active_fit_start", estimator=self, datamodule=active_datamodule, output=outputs)

        for _ in pbar:
            output = self.round_loop(
                active_datamodule=active_datamodule,
                query_size=query_size,
                num_epochs=num_epochs,
                loss_fn=loss_fn,
                loss_fn_kwargs=loss_fn_kwargs,
                learning_rate=learning_rate,
                optimizer=optimizer,
                optimizer_kwargs=optimizer_kwargs,
                scheduler=scheduler,
                scheduler_kwargs=scheduler_kwargs,
                log_interval=log_interval,
                limit_train_batches=limit_train_batches,
                limit_validation_batches=limit_validation_batches,
                limit_test_batches=limit_test_batches,
                dry_run=dry_run,
            )
            outputs.append(output)

            # label data
            if output.query.indices is not None:
                # hook
                self.fabric.call("on_label_start", estimator=self, datamodule=active_datamodule, output=outputs)

                active_datamodule.label(indices=output.query.indices, round_idx=output.round_idx, val_perc=val_perc)

                # hook
                self.fabric.call("on_label_end", estimator=self, datamodule=active_datamodule, output=outputs)

            if reinit_model:
                self.load_state_dict(model_cache_dir)

            # update counter
            self.counter.increment_rounds()

            # check stopping conditions
            if self._is_done(dry_run):
                break

        # hook
        self.fabric.call("on_active_fit_end", estimator=self, datamodule=active_datamodule, output=outputs)

        return outputs

    def round_loop(
        self,
        active_datamodule: ActiveDataModule,
        query_size: int,
        num_epochs: Optional[int] = 3,
        loss_fn: Optional[Union[str, torch.nn.Module, Callable]] = None,
        loss_fn_kwargs: Optional[Dict] = None,
        learning_rate: float = 0.001,
        optimizer: str = "adamw",
        optimizer_kwargs: Optional[Dict] = None,
        scheduler: Optional[str] = None,
        scheduler_kwargs: Optional[Dict] = None,
        log_interval: int = 1,
        limit_train_batches: Optional[int] = None,
        limit_validation_batches: Optional[int] = None,
        limit_test_batches: Optional[int] = None,
        dry_run: Optional[bool] = False,
        **kwargs,
    ) -> RoundOutput:
        output = RoundOutput()

        # hook
        self.fabric.call("on_round_start", estimator=self, datamodule=active_datamodule, output=output)

        # fit model on the available data
        if active_datamodule.has_labelled_data:
            output.fit = self.fit(
                train_loader=active_datamodule.train_loader(),
                validation_loader=active_datamodule.validation_loader(),
                num_epochs=num_epochs,
                loss_fn=loss_fn,
                loss_fn_kwargs=loss_fn_kwargs,
                learning_rate=learning_rate,
                optimizer=optimizer,
                optimizer_kwargs=optimizer_kwargs,
                scheduler=scheduler,
                scheduler_kwargs=scheduler_kwargs,
                log_interval=log_interval,
                limit_train_batches=limit_train_batches,
                limit_validation_batches=limit_validation_batches,
                dry_run=dry_run,
            )

        # test model
        if active_datamodule.has_test_data:
            output.test = self.test(
                test_loader=active_datamodule.test_loader(),
                loss_fn=loss_fn,
                loss_fn_kwargs=loss_fn_kwargs,
                log_interval=log_interval,
                dry_run=dry_run,
                limit_batches=limit_test_batches,
            )

        # query indices to annotate
        if active_datamodule.pool_size > query_size:
            output.query = self.query(
                active_datamodule=active_datamodule,
                query_size=query_size,
                dry_run=dry_run,
                log_interval=log_interval,
                **kwargs,
            )

        # hook
        self.fabric.call("on_round_end", estimator=self, datamodule=active_datamodule, output=output)

        return output

    """
    Query loop
    """

    def query(self, active_datamodule: ActiveDataModule, query_size: int, **kwargs) -> QueryOutput:
        # configure dataloaders
        loader = self.get_pool_loader(active_datamodule=active_datamodule)
        loader = self.configure_dataloader(loader)

        # setup model with fabric
        model = self.fabric.setup(self.model)

        # hook
        self.fabric.call("on_query_start", estimator=self, model=model)

        output = self.query_loop(model, loader, query_size, **kwargs)

        # hook
        self.fabric.call("on_query_end", estimator=self, model=model, output=output)

    def query_loop(self, model: _FabricModule, loader: _FabricDataLoader, query_size: int, **kwargs) -> QueryOutput:
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
        self.fabric.save(self.model.state_dict(), cache_dir / "state_dict.pt")

    def load_state_dict(self, cache_dir: Union[str, Path]) -> None:
        cache_dir = Path(cache_dir)
        self.model.load_state_dict(self.fabric.load(cache_dir / "state_dict.pt"))

    def get_pool_loader(self, active_datamodule: ActiveDataModule) -> DataLoader:
        return active_datamodule.pool_loader()

    """
    Utilities
    """

    def _get_round_progress_bar(self, num_rounds: int) -> tqdm:
        return trange(num_rounds, desc="Completed labelling rounds", dynamic_ncols=True, leave=True)

    def _get_epoch_progress_bar(self, num_epochs: int) -> tqdm:
        return trange(num_epochs, desc="Completed epochs", dynamic_ncols=True, leave=False)
