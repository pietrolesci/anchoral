import os
from typing import Any, List, Optional, Union

import torch
from lightning.fabric import Fabric
from lightning.fabric.accelerators.accelerator import Accelerator
from lightning.fabric.connector import _PLUGIN_INPUT, _PRECISION_INPUT
from lightning.fabric.loggers import Logger
from lightning.fabric.strategies import Strategy
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from tqdm.auto import tqdm, trange

from src.enums import RunningStage
from src.hooks import HooksMixin
from src.types import BATCH_OUTPUT


class Trainer(HooksMixin):
    def __init__(
        self,
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
        super().__init__()
        self.fabric = Fabric(
            accelerator=accelerator,
            strategy=strategy,
            devices=devices,
            num_nodes=num_nodes,
            precision=precision,
            plugins=plugins,
            callbacks=callbacks,
            loggers=loggers,
        )
        self._init_deterministic(deterministic)

    def _init_deterministic(self, deterministic: bool) -> None:
        # NOTE: taken from the lightning Trainer
        torch.use_deterministic_algorithms(deterministic)
        if deterministic:
            # fixing non-deterministic part of horovod
            # https://github.com/Lightning-AI/lightning/pull/1572/files#r420279383
            os.environ["HOROVOD_FUSION_THRESHOLD"] = "0"

            # https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    def fit(
        self,
        model: torch.nn.Module,
        optimizer: Optimizer,
        scheduler: _LRScheduler,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        hparams,
    ):
        """Runs full training and validation."""

        # setup dataloaders, model, and optimizer with fabric
        train_loader = self.fabric.setup_dataloaders(
            train_loader,
            replace_sampler=False,
            move_to_device=False,
        )
        if val_loader is not None:
            val_loader = self.fabric.setup_dataloaders(
                val_loader,
                replace_sampler=False,
                move_to_device=False,
            )

        model, optimizer = self.fabric.setup(model, optimizer)

        # Training loop over epochs
        for epoch in trange(hparams.num_epochs):

            self.train_epoch_loop(model, train_loader, optimizer, scheduler, epoch, hparams)

            # and maybe validates at the end of each epoch
            if val_loader is not None:
                self.eval_epoch_loop(model, val_loader, RunningStage.VALIDATION, hparams)

    def train_epoch_loop(
        self,
        model: torch.nn.Module,
        optimizer: Optimizer,
        scheduler: _LRScheduler,
        train_loader: DataLoader,
        epoch: int,
        hparams,
    ):
        """Runs over an entire dataloader."""

        model.train()

        self.on_train_epoch_start(model)

        pbar = tqdm(train_loader, desc=f"Epoch: {epoch}", dynamic_ncols=True)
        for batch_idx, batch in enumerate(pbar):

            batch = self.transfer_to_device(batch)

            self.on_train_batch_start(model, batch, batch_idx)

            output = self.train_batch_loop(model, batch, batch_idx, optimizer, scheduler)

            self.on_train_batch_end(model, output, batch, batch_idx)

            if (batch_idx == 0) or ((batch_idx + 1) % hparams.log_interval == 0):
                pbar.set_postfix(loss=round(output.loss.item(), ndigits=4))

            if hparams.dry_run:
                break

        self.on_train_epoch_end(model)

    def train_batch_loop(
        self,
        model: torch.nn.Module,
        batch: Any,
        batch_idx: int,
        optimizer: Optimizer,
        scheduler: _LRScheduler,
    ) -> BATCH_OUTPUT:
        """Runs over a single batch of data."""

        # zero_grad
        optimizer.zero_grad()

        # compute loss
        output = self.training_step(model, batch, batch_idx)
        loss = output if isinstance(output, torch.Tensor) else output["loss"]

        # compute gradients
        self.fabric.backward(loss)  # instead of loss.backward()

        # update parameters
        optimizer.step()

        # update scheduler
        if scheduler is not None:
            scheduler.step()

        return output

    def eval_epoch_loop(self, model: torch.nn.Module, eval_loader: DataLoader, stage: RunningStage, hparams):
        """Runs over an entire evaluation dataloader."""

        getattr(self, f"on_{stage}_epoch_start")(model)

        model.eval()

        pbar = tqdm(eval_loader, desc=f"{stage.title()}", dynamic_ncols=True)
        with torch.inference_mode():
            for batch_idx, batch in enumerate(pbar):
                batch = self.transfer_to_device(batch)

                getattr(self, f"on_{stage}_batch_start")(model)

                output = getattr(self, f"{stage}_step")(batch, batch_idx)

                getattr(self, f"on_{stage}_batch_end")(model)

                if hparams.dry_run:
                    break

        getattr(self, f"on_{stage}_epoch_end")(model)

        return output

    def validate(self, model: torch.nn.Module, val_loader: DataLoader, hparams):
        # register dataloader and model with fabric
        val_loader = self.fabric.setup_dataloaders(val_loader)
        model = self.fabric.setup(model)
        return self.eval_epoch_loop(model, val_loader, RunningStage.VALIDATION, hparams)

    def test(self, model: torch.nn.Module, test_loader: DataLoader, hparams):
        # register dataloader and model with fabric
        test_loader = self.fabric.setup_dataloaders(test_loader)
        model = self.fabric.setup(model)
        return self.eval_epoch_loop(model, test_loader, RunningStage.TEST, hparams)

    def transfer_to_device(self, batch: Any) -> Any:
        batch = self.fabric.to_device(batch)
        return batch

    def training_step(self, model: torch.nn.Module, batch: Any, batch_idx: int):
        pass

    def validation_step(self, model: torch.nn.Module, batch: Any, batch_idx: int):
        pass

    def test_step(self, model: torch.nn.Module, batch: Any, batch_idx: int):
        pass
