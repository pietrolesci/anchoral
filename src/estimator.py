import os
from typing import Any, Dict, List, Optional, Union

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
from src.registries import OPTIMIZER_REGISTRY, SCHEDULER_REGISTRY
from src.types import BATCH_OUTPUT


class Estimator:
    def __init__(
        self,
        model: torch.nn.Module,
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
        self.model = model

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
        train_loader: DataLoader,
        validation_loader: Optional[DataLoader] = None,
        learning_rate: Optional[float] = 0.001,
        optimizer: Union[str, Optimizer] = "adamw",
        optimizer_kwargs: Optional[Dict] = None,
        scheduler: Optional[Union[str, _LRScheduler]] = None,
        scheduler_kwargs: Optional[Dict] = None,
        **kwargs,
    ):
        """Runs full training and validation."""

        # define optimizer and scheduler
        assert optimizer is not None, ValueError("You must provide an optimizer.")
        if isinstance(optimizer, str):
            optimizer_kwargs = optimizer_kwargs or {}
            optimizer = self.configure_optimizer(optimizer, learning_rate, **optimizer_kwargs)

        if scheduler is not None and isinstance(scheduler, str):
            num_training_steps = self._compute_num_training_steps(train_loader)
            scheduler_kwargs = scheduler_kwargs or {}
            scheduler = self.configure_scheduler(
                scheduler, optimizer, num_training_steps=num_training_steps, **scheduler_kwargs
            )

        # setup dataloaders, model, and optimizer with fabric
        train_loader = self.fabric.setup_dataloaders(
            train_loader,
            replace_sampler=False,
            move_to_device=False,
        )
        if validation_loader is not None:
            validation_loader = self.fabric.setup_dataloaders(
                validation_loader,
                replace_sampler=False,
                move_to_device=False,
            )
        model, optimizer = self.fabric.setup(self.model, optimizer)

        # Training loop over epochs
        for epoch in trange(kwargs.get("num_epochs", 3), desc="Completed epochs"):

            self.train_epoch_loop(model, train_loader, optimizer, scheduler, epoch=epoch, **kwargs)

            # and maybe validates at the end of each epoch
            if validation_loader is not None:
                self.eval_epoch_loop(model, validation_loader, RunningStage.VALIDATION, **kwargs)

    def train_epoch_loop(
        self,
        model: torch.nn.Module,
        train_loader: DataLoader,
        optimizer: Optimizer,
        scheduler: _LRScheduler,
        **kwargs,
    ):
        """Runs over an entire dataloader."""

        model.train()

        pbar = tqdm(train_loader, desc=f"Epoch {kwargs.get('epoch', '')}".strip(), dynamic_ncols=True, leave=False)
        for batch_idx, batch in enumerate(pbar):

            batch = self.transfer_to_device(batch)

            output = self.train_batch_loop(model, batch, batch_idx, optimizer, scheduler)

            if (batch_idx == 0) or ((batch_idx + 1) % kwargs.get("log_interval", 1) == 0):
                pbar.set_postfix(loss=round(output["loss"].item(), ndigits=4))

            if kwargs.get("dry_run", False) or kwargs.get("limit_train_batches", float("inf")) <= batch_idx:
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

    def eval_epoch_loop(self, model: torch.nn.Module, eval_loader: DataLoader, stage: RunningStage, **kwargs):
        """Runs over an entire evaluation dataloader."""

        model.eval()

        pbar = tqdm(eval_loader, desc=f"{stage.title()}", dynamic_ncols=True, leave=False)
        with torch.inference_mode():
            for batch_idx, batch in enumerate(pbar):
                batch = self.transfer_to_device(batch)

                output = getattr(self, f"{stage}_step")(model, batch, batch_idx)

                if kwargs.get("dry_run", False) or kwargs.get(f"limit_{stage}_batches", float("inf")) <= batch_idx:
                    break

        return output

    def validate(self, validation_loader: DataLoader, **kwargs):
        # register dataloader and model with fabric
        model = self.fabric.setup(self.model)
        validation_loader = self.fabric.setup_dataloaders(validation_loader)
        return self.eval_epoch_loop(model, validation_loader, RunningStage.VALIDATION, **kwargs)

    def test(self, test_loader: DataLoader, **kwargs):
        # register dataloader and model with fabric
        model = self.fabric.setup(self.model)
        test_loader = self.fabric.setup_dataloaders(test_loader)
        return self.eval_epoch_loop(model, test_loader, RunningStage.TEST, **kwargs)

    def transfer_to_device(self, batch: Any) -> Any:
        batch = self.fabric.to_device(batch)
        return batch

    def training_step(self, model: torch.nn.Module, batch: Any, batch_idx: int):
        pass

    def validation_step(self, model: torch.nn.Module, batch: Any, batch_idx: int):
        pass

    def test_step(self, model: torch.nn.Module, batch: Any, batch_idx: int):
        pass

    def configure_optimizer(self, optimizer: str, learning_rate: float, **optimizer_kwargs) -> Optimizer:
        """Handled optimizer and scheduler configuration."""

        # collect optimizer kwargs
        no_decay, weight_decay = optimizer_kwargs.get("no_decay", None), optimizer_kwargs.get("weight_decay", None)

        # filter parameters to optimize
        if no_decay is not None and (weight_decay is not None and weight_decay > 0.0):
            params = [
                {
                    "params": [
                        p
                        for n, p in self.model.named_parameters()
                        if not any(nd in n for nd in no_decay) and p.requires_grad
                    ],
                    "weight_decay": weight_decay,
                },
                {
                    "params": [
                        p
                        for n, p in self.model.named_parameters()
                        if any(nd in n for nd in no_decay) and p.requires_grad
                    ],
                    "weight_decay": 0.0,
                },
            ]
        else:
            params = filter(lambda p: p.requires_grad, self.model.parameters())

        # instantiate optimizer
        optimizer = OPTIMIZER_REGISTRY.get(optimizer)(params, lr=learning_rate, **optimizer_kwargs)

        return optimizer

    def configure_scheduler(
        self, scheduler: str, optimizer: Optimizer, num_training_steps: int, **scheduler_kwargs
    ) -> _LRScheduler:
        num_warmup_steps = scheduler_kwargs.get("num_warmup_steps", None)
        if num_warmup_steps is not None and isinstance(num_warmup_steps, float):
            num_warmup_steps *= num_training_steps

        scheduler = SCHEDULER_REGISTRY.get(scheduler)(
            optimizer, num_training_steps=num_training_steps, num_warmup_steps=num_warmup_steps
        )

        return scheduler

    @staticmethod
    def _compute_num_training_steps(train_loader: DataLoader) -> int:
        # FIXME: when accumulate batches is added
        return len(train_loader)


# class ActiveTrainer(Trainer):
#     def active_learning_loop(self, query_strategy, model, hparams):
#         for round in trange(hparams.num_rounds):
#             train_output, val_output = self.fit_loop()
#             test_output = self.test_loop()
#             query_output = self.strategy.query()

#         return

#     def transfer_to_device(self, batch):
#         data_on_cpu = batch.pop("data_on_cpu", None)

#         # transfer the rest on gpu
#         batch = self.fabric.to_device(batch)

#         # add the columns on cpu to the batch
#         if data_on_cpu is not None:
#             batch["data_on_cpu"] = data_on_cpu

#         return batch
