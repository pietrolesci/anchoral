from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional, Union

import torch
from lightning.fabric.wrappers import _FabricDataLoader, _FabricModule
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.energizer.active_learning.data import ActiveDataModule
from src.energizer.containers import ActiveFitOutput, QueryOutput, RoundOutput
from src.energizer.enums import RunningStage
from src.energizer.estimator import Estimator
from src.energizer.progress_trackers import ActiveProgressTracker
from src.energizer.utilities import get_hparams


class ActiveEstimator(Estimator):
    _counter: ActiveProgressTracker = None

    @property
    def progress_tracker(self) -> ActiveProgressTracker:
        if self._counter is None:
            self._counter = ActiveProgressTracker()
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
        **kwargs,
    ) -> ActiveFitOutput:
        # get passed hyper-parameters
        hparams = get_hparams()
        outputs = ActiveFitOutput(hparams=hparams)

        if reinit_model:
            self.save_state_dict(model_cache_dir)

        # reset progress_tracker
        self.progress_tracker.reset()

        pbar = self._get_round_progress_bar(num_rounds, **kwargs)  # +1 because on round 0 we do not query

        # hook
        self.fabric.call("on_active_fit_start", estimator=self, datamodule=active_datamodule, output=outputs)

        while self.progress_tracker.num_rounds < num_rounds:
            output = self.round_loop(
                active_datamodule=active_datamodule,
                query_size=query_size,
                val_perc=val_perc,
                num_epochs=num_epochs,
                loss_fn=loss_fn,
                loss_fn_kwargs=loss_fn_kwargs,
                learning_rate=learning_rate,
                optimizer=optimizer,
                optimizer_kwargs=optimizer_kwargs,
                scheduler=scheduler,
                scheduler_kwargs=scheduler_kwargs,
                **kwargs,
            )
            self.progress_tracker.increment_rounds()

            outputs.append(output)

            if reinit_model:
                self.load_state_dict(model_cache_dir)

            # do not increase progress bar on the warm-up round
            if isinstance(pbar, tqdm) and self.progress_tracker.num_rounds > 0:
                pbar.update(1)

        # hook
        self.fabric.call("on_active_fit_end", estimator=self, datamodule=active_datamodule, output=outputs)

        return outputs

    def round_loop(
        self,
        active_datamodule: ActiveDataModule,
        query_size: int,
        val_perc: float,
        num_epochs: Optional[int] = 3,
        loss_fn: Optional[Union[str, torch.nn.Module, Callable]] = None,
        loss_fn_kwargs: Optional[Dict] = None,
        learning_rate: float = 0.001,
        optimizer: str = "adamw",
        optimizer_kwargs: Optional[Dict] = None,
        scheduler: Optional[str] = None,
        scheduler_kwargs: Optional[Dict] = None,
        **kwargs,
    ) -> RoundOutput:
        output = RoundOutput()

        # hook
        self.fabric.call("on_round_start", estimator=self, datamodule=active_datamodule, output=output)

        # query indices to annotate, skip first round
        # do not annotate on the warm-up round
        if active_datamodule.pool_size > query_size and self.progress_tracker.num_rounds >= 0:
            output.query = self.query(active_datamodule=active_datamodule, query_size=query_size, **kwargs)
            self.progress_tracker.increment_total_batches(RunningStage.POOL)

            # hook
            self.fabric.call("on_label_start", estimator=self, datamodule=active_datamodule, output=output)
            active_datamodule.label(
                indices=output.query.indices, round_idx=self.progress_tracker.num_rounds, val_perc=val_perc
            )
            # hook
            self.fabric.call("on_label_end", estimator=self, datamodule=active_datamodule, output=output)

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
                **kwargs,
            )
            self.progress_tracker.increment_total_epochs()
            self.progress_tracker.increment_total_batches(RunningStage.TRAIN)
            self.progress_tracker.increment_total_batches(RunningStage.VALIDATION)

        # test model
        if active_datamodule.has_test_data:
            output.test = self.test(
                test_loader=active_datamodule.test_loader(),
                loss_fn=loss_fn,
                loss_fn_kwargs=loss_fn_kwargs,
                **kwargs,
            )
            self.progress_tracker.increment_total_batches(RunningStage.TEST)

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

        return output

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

    def _get_round_progress_bar(self, num_rounds: int, **kwargs) -> Union[tqdm, Iterable]:
        # check if progress bar is disabled
        progress_bar = kwargs.get("progress_bar", True)
        if not progress_bar:
            return range(num_rounds)

        return tqdm(total=num_rounds, desc="Completed rounds", dynamic_ncols=True, leave=True)

    def _get_epoch_progress_bar(self, *args, **kwargs) -> Optional[tqdm]:
        pbar = super()._get_epoch_progress_bar(*args, **kwargs)
        # remove the epoch progress_tracker progress bar
        if isinstance(pbar, tqdm):
            pbar.leave = False
        return pbar
