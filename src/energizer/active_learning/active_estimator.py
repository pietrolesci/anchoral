from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
from lightning.fabric.wrappers import _FabricModule
from torch.utils.data import DataLoader

from src.energizer.active_learning.data import ActiveDataModule
from src.energizer.active_learning.progress_trackers import ActiveProgressTracker
from src.energizer.enums import RunningStage
from src.energizer.estimator import Estimator, FitEpochOutput
from src.energizer.types import EPOCH_OUTPUT, ROUND_OUTPUT


@dataclass
class RoundOutput:
    fit: List[FitEpochOutput] = None
    test: EPOCH_OUTPUT = None
    indices: List[int] = None


class ActiveEstimator(Estimator):
    _progress_tracker: ActiveProgressTracker = None

    @property
    def progress_tracker(self) -> ActiveProgressTracker:
        if self._progress_tracker is None:
            self._progress_tracker = ActiveProgressTracker()
        return self._progress_tracker

    """
    Active learning loop
    """

    def active_fit(
        self,
        active_datamodule: ActiveDataModule,
        max_rounds: int,
        query_size: int,
        validation_perc: Optional[float] = None,
        max_budget: Optional[int] = None,
        validation_sampling: Optional[str] = None,
        reinit_model: bool = True,
        max_epochs: Optional[int] = 3,
        min_steps: Optional[int] = None,
        learning_rate: float = 0.001,
        optimizer: str = "adamw",
        optimizer_kwargs: Optional[Dict] = None,
        scheduler: Optional[str] = None,
        scheduler_kwargs: Optional[Dict] = None,
        model_cache_dir: Optional[Union[str, Path]] = ".model_cache",
        **kwargs,
    ) -> Any:
        if reinit_model:
            self.save_state_dict(model_cache_dir)

        # configure progress tracking
        self.progress_tracker.setup(
            max_rounds=max_rounds,
            max_budget=min(active_datamodule.pool_size, max_budget or float("Inf")),
            initial_budget=active_datamodule.total_labelled_size,
            query_size=query_size,
            has_pool=getattr(self, "pool_step", None) is not None,
            has_validation=active_datamodule.validation_loader() is not None or validation_perc,
            has_test=active_datamodule.test_loader() is not None,
        )

        # call hook
        self.fabric.call("on_active_fit_start", estimator=self, datamodule=active_datamodule)

        output = []
        while not self.progress_tracker.is_active_fit_done():

            if reinit_model:
                self.load_state_dict(model_cache_dir)

            self.fabric.call("on_round_start", estimator=self, datamodule=active_datamodule)

            out = self.run_round(
                active_datamodule=active_datamodule,
                query_size=query_size,
                validation_perc=validation_perc,
                validation_sampling=validation_sampling,
                max_epochs=max_epochs,
                min_steps=min_steps,
                learning_rate=learning_rate,
                optimizer=optimizer,
                optimizer_kwargs=optimizer_kwargs,
                scheduler=scheduler,
                scheduler_kwargs=scheduler_kwargs,
                **kwargs,
            )

            self.fabric.call("on_round_end", estimator=self, datamodule=active_datamodule, output=out)

            output.append(out)

            # update progress
            self.progress_tracker.increment_round()
            assert (
                self.progress_tracker.budget_tracker.total == active_datamodule.total_labelled_size
            ), f"{self.progress_tracker.budget_tracker.total} == {active_datamodule.total_labelled_size}"

        if not self.progress_tracker.global_round > 0:
            raise ValueError("You did not run any labellng. Perhaps change your `max_budget` or `max_rounds`.")

        output = self.active_fit_end(output)

        # call hook
        self.fabric.call("on_active_fit_end", estimator=self, datamodule=active_datamodule, output=output)

        self.progress_tracker.end_active_fit()

        return output

    def run_round(
        self,
        active_datamodule: ActiveDataModule,
        query_size: int,
        validation_perc: float,
        validation_sampling: Optional[str],
        max_epochs: Optional[int],
        min_steps: Optional[int],
        learning_rate: float,
        optimizer: str,
        optimizer_kwargs: Optional[Dict],
        scheduler: Optional[str],
        scheduler_kwargs: Optional[Dict],
        **kwargs,
    ) -> ROUND_OUTPUT:
        output = RoundOutput()

        self.progress_tracker.setup_round_tracking(
            # fit
            max_epochs=max_epochs,
            min_steps=min_steps,
            num_train_batches=len(active_datamodule.train_loader() or []),
            num_validation_batches=len(active_datamodule.validation_loader() or []),
            limit_train_batches=kwargs.get("limit_train_batches"),
            limit_validation_batches=kwargs.get("limit_validation_batches"),
            validation_interval=kwargs.get("validation_interval"),
            # test
            num_test_batches=len(active_datamodule.test_loader() or []),
            limit_test_batches=kwargs.get("limit_test_batches"),
            # pool
            num_pool_batches=len(active_datamodule.pool_loader() or []),
            limit_pool_batches=kwargs.get("limit_pool_batches"),
        )

        train_loader = self.configure_dataloader(active_datamodule.train_loader())
        validation_loader = self.configure_dataloader(active_datamodule.validation_loader())
        test_loader = self.configure_dataloader(active_datamodule.test_loader())
        optimizer = self.configure_optimizer(optimizer, learning_rate, optimizer_kwargs)
        scheduler = self.configure_scheduler(scheduler, optimizer, scheduler_kwargs)
        model, optimizer = self.fabric.setup(self.model, optimizer)

        # fit
        if active_datamodule.has_labelled_data:
            output.fit = self.run_fit(model, train_loader, validation_loader, optimizer, scheduler)

        # test
        if active_datamodule.has_test_data:
            output.test = self.run_evaluation(model, test_loader, RunningStage.TEST)

        # query and label
        if active_datamodule.pool_size > query_size:
            self.fabric.call("on_query_start", estimator=self, model=model)
            output.indices = self.run_query(model, active_datamodule, query_size)
            self.fabric.call("on_query_end", estimator=self, model=model, output=output)

            self.fabric.call("on_label_start", estimator=self, datamodule=active_datamodule)
            active_datamodule.label(
                indices=output.indices,
                round_idx=self.progress_tracker.global_round,
                validation_perc=validation_perc,
                validation_sampling=validation_sampling,
            )
            self.fabric.call("on_label_end", estimator=self, datamodule=active_datamodule)

        # method to possibly aggregate
        output = self.round_epoch_end(output, active_datamodule)

        return output

    """
    Query loop
    """

    def active_fit_end(self, output: List[ROUND_OUTPUT]) -> Any:
        return output

    def round_epoch_end(self, output: RoundOutput, datamodule: ActiveDataModule) -> ROUND_OUTPUT:
        return output

    def run_query(self, model: _FabricModule, active_datamodule: ActiveDataModule, query_size: int) -> List[int]:
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

    def get_pool_loader(self, active_datamodule: ActiveDataModule) -> DataLoader:
        return active_datamodule.pool_loader()

    # def replay_active_fit(
    #     self,
    #     active_datamodule: ActiveDataModule,
    #     max_epochs: Optional[int] = 3,
    #     min_steps: Optional[int] = None,
    #     learning_rate: float = 0.001,
    #     optimizer: str = "adamw",
    #     optimizer_kwargs: Optional[Dict] = None,
    #     scheduler: Optional[str] = None,
    #     scheduler_kwargs: Optional[Dict] = None,
    #     **kwargs,
    # ) -> List[FitEpochOutput]:

    #     num_rounds = active_datamodule.

    #     for round in rounds:

    #         train_loader, validation_loader, test_loader = ...

    #         self.progress_tracker.setup_tracking(
    #             max_epochs=max_epochs,
    #             min_steps=min_steps,
    #             num_train_batches=len(active_datamodule.train_loader()) if active_datamodule.train_loader() else 0,
    #             num_validation_batches=len(active_datamodule.validation_loader())
    #             if active_datamodule.validation_loader()
    #             else 0,
    #             num_test_batches=len(active_datamodule.test_loader()) if active_datamodule.test_loader() else 0,
    #             num_pool_batches=len(active_datamodule.pool_loader()) if active_datamodule.pool_loader() else 0,
    #             limit_train_batches=kwargs.get("limit_train_batches"),
    #             limit_validation_batches=kwargs.get("limit_validation_batches"),
    #             limit_test_batches=kwargs.get("limit_test_batches"),
    #             validation_interval=kwargs.get("validation_interval"),
    #         )

    #         train_loader = self.configure_dataloader(active_datamodule.train_loader())
    #         validation_loader = self.configure_dataloader(active_datamodule.validation_loader())
    #         test_loader = self.configure_dataloader(active_datamodule.test_loader())
    #         optimizer = self.configure_optimizer(optimizer, learning_rate, optimizer_kwargs)
    #         scheduler = self.configure_scheduler(scheduler, optimizer, scheduler_kwargs)
    #         model, optimizer = self.fabric.setup(self.model, optimizer)

    #         # fit
    #         output.fit = self.run_fit(model, train_loader, validation_loader, optimizer, scheduler)

    #         # test
    #         if active_datamodule.has_test_data:
    #             output.test = self.run_evaluation(model, test_loader, RunningStage.TEST)
