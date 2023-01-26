import os
import warnings
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
from torchmetrics import Metric, MetricCollection
from tqdm.auto import tqdm, trange

from src.enums import RunningStage
from src.registries import OPTIMIZER_REGISTRY, SCHEDULER_REGISTRY
from src.types import BATCH_OUTPUT, EpochOutput, EvaluationOutput, FitOutput
from src.utilities import get_hparams

warnings.filterwarnings("ignore", message="The ``compute`` method of")
import inspect


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
        num_epochs: int = 3,
        learning_rate: float = 0.001,
        optimizer: str = "adamw",
        optimizer_kwargs: Optional[Dict] = None,
        scheduler: Optional[str] = None,
        scheduler_kwargs: Optional[Dict] = None,
        log_interval: int = 1,
        dry_run: Optional[bool] = None,
        limit_train_batches: Optional[int] = None,
        limit_validation_batches: Optional[int] = None,
    ):
        """Runs full training and validation."""

        # get passed hyper-parameters
        hparams = get_hparams()

        # define optimizer and scheduler
        assert optimizer is not None, ValueError("You must provide an optimizer.")
        optimizer_kwargs = optimizer_kwargs or {}
        optimizer = self.configure_optimizer(optimizer, learning_rate, **optimizer_kwargs)
        if scheduler is not None:
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

        outputs = FitOutput(hparams=hparams)

        # Training loop over epochs
        for epoch_idx in trange(num_epochs, desc="Completed epochs"):

            # run train epoch
            output = self.train_epoch_loop(
                model,
                train_loader,
                optimizer,
                scheduler,
                epoch_idx=epoch_idx,
                log_interval=log_interval,
                dry_run=dry_run,
                limit_batches=limit_train_batches,
            )
            outputs.append(output, RunningStage.TRAIN)

            # and maybe validates at the end of each epoch
            if validation_loader is not None:
                output = self.eval_epoch_loop(
                    model,
                    validation_loader,
                    RunningStage.VALIDATION,
                    dry_run=dry_run,
                    limit_batches=limit_validation_batches,
                    epoch_idx=epoch_idx,
                )
                outputs.append(output, RunningStage.VALIDATION)

        return outputs

    def train_epoch_loop(
        self,
        model: torch.nn.Module,
        train_loader: DataLoader,
        optimizer: Optimizer,
        scheduler: _LRScheduler,
        epoch_idx: int,
        log_interval: int = 1,
        dry_run: Optional[bool] = None,
        limit_batches: Optional[int] = None,
    ) -> EpochOutput:
        """Runs over an entire dataloader."""

        model.train()

        # call callback hook
        self.fabric.call("on_train_epoch_start", model=model)

        # define metrics
        metrics = self.configure_metrics(RunningStage.TRAIN).to(self.fabric.device)

        outputs = EpochOutput(epoch=epoch_idx)

        pbar = tqdm(train_loader, desc=f"Epoch {epoch_idx}".strip(), dynamic_ncols=True, leave=False)
        for batch_idx, batch in enumerate(pbar):

            # put batch on correct device
            batch = self.transfer_to_device(batch)

            # call callback hook
            self.fabric.call("on_train_batch_start", model=model, batch=batch, batch_idx=batch_idx)

            # run model on batch
            output = self.train_batch_loop(model, batch, batch_idx, optimizer, scheduler, metrics)

            # call callback hook
            self.fabric.call("on_train_batch_end", model=model, output=output, batch=batch, batch_idx=batch_idx)

            # record output
            outputs.append(output)

            # update progress
            if (batch_idx == 0) or ((batch_idx + 1) % log_interval == 0):
                pbar.set_postfix(loss=round(output["loss"].item(), ndigits=4))

            # check stopping conditions
            if self._is_done(batch_idx, dry_run, limit_batches):
                break

        # call callback hook
        self.fabric.call("on_train_epoch_end", model=model)

        # aggregate metric over epoch
        if metrics is not None:
            outputs.add_metrics(metrics.compute())

        return outputs

    def train_batch_loop(
        self,
        model: torch.nn.Module,
        batch: Any,
        batch_idx: int,
        optimizer: Optimizer,
        scheduler: _LRScheduler,
        metrics: Any,
    ) -> BATCH_OUTPUT:
        """Runs over a single batch of data."""

        # zero_grad
        optimizer.zero_grad()

        # compute loss
        output = self.training_step(model, batch, batch_idx, metrics)
        loss = output if isinstance(output, torch.Tensor) else output["loss"]

        # compute gradients
        self.fabric.backward(loss)  # instead of loss.backward()

        # update parameters
        optimizer.step()

        # update scheduler
        if scheduler is not None:
            scheduler.step()

        return output

    def eval_epoch_loop(
        self, model: torch.nn.Module, eval_loader: DataLoader, stage: RunningStage, **kwargs
    ) -> EpochOutput:
        """Runs over an entire evaluation dataloader."""

        model.eval()

        # call callback hook
        self.fabric.call(f"on_{stage}_epoch_start", model=model)

        # define metrics
        metrics = self.configure_metrics(stage).to(self.fabric.device)

        outputs = EpochOutput(epoch=kwargs.get("epoch_idx", None))

        pbar = tqdm(eval_loader, desc=f"{stage.title()}", dynamic_ncols=True, leave=False)
        with torch.inference_mode():
            for batch_idx, batch in enumerate(pbar):
                batch = self.transfer_to_device(batch)

                # call callback hook
                self.fabric.call(f"on_{stage}_batch_start", model=model, batch=batch, batch_idx=batch_idx)

                output = getattr(self, f"{stage}_step")(model, batch, batch_idx, metrics)

                # call callback hook
                self.fabric.call(f"on_{stage}_batch_end", model=model, output=output, batch=batch, batch_idx=batch_idx)

                # record output
                outputs.append(output)

                # check stopping conditions
                if self._is_done(batch_idx, kwargs.get("dry_run", None), kwargs.get("limit_batches", None)):
                    break

        # call callback hook
        self.fabric.call(f"on_{stage}_epoch_end", model=model)

        # aggregate metric over epoch
        if metrics is not None:
            outputs.add_metrics(metrics.compute())

        return outputs

    def validate(
        self,
        validation_loader: DataLoader,
        dry_run: Optional[bool] = None,
        limit_batches: Optional[int] = None,
    ) -> EpochOutput:

        # get passed hyper-parameters
        hparams = get_hparams()

        # register dataloader and model with fabric
        model = self.fabric.setup(self.model)
        validation_loader = self.fabric.setup_dataloaders(validation_loader)

        # run validation
        output = self.eval_epoch_loop(
            model, validation_loader, stage=RunningStage.VALIDATION, dry_run=dry_run, limit_batches=limit_batches
        )

        return EvaluationOutput(hparams=hparams, output=output)

    def test(
        self,
        test_loader: DataLoader,
        dry_run: Optional[bool] = None,
        limit_batches: Optional[int] = None,
    ) -> EpochOutput:

        # get passed hyper-parameters
        hparams = get_hparams()

        # register dataloader and model with fabric
        model = self.fabric.setup(self.model)
        test_loader = self.fabric.setup_dataloaders(test_loader)

        # run testing
        output = self.eval_epoch_loop(
            model, test_loader, stage=RunningStage.TEST, dry_run=dry_run, limit_batches=limit_batches
        )

        return EvaluationOutput(hparams=hparams, output=output)

    def transfer_to_device(self, batch: Any) -> Any:
        batch = self.fabric.to_device(batch)
        return batch

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

        scheduler_fn = SCHEDULER_REGISTRY.get(scheduler)
        params = list(inspect.signature(scheduler_fn).parameters.keys())

        if "num_training_steps" in params:
            scheduler_kwargs["num_training_steps"] = num_training_steps
        if "num_warmup_steps" in params:
            scheduler_kwargs["num_warmup_steps"] = num_warmup_steps

        scheduler = scheduler_fn(optimizer, **scheduler_kwargs)

        return scheduler

    def configure_metrics(self, stage: Optional[RunningStage] = None) -> Union[MetricCollection, Metric, None]:
        return

    def training_step(
        self, model: torch.nn.Module, batch: Any, batch_idx: int, metrics: Optional[Any] = None
    ) -> BATCH_OUTPUT:
        raise NotImplementedError

    def validation_step(
        self, model: torch.nn.Module, batch: Any, batch_idx: int, metrics: Optional[Any] = None
    ) -> BATCH_OUTPUT:
        raise NotImplementedError

    def test_step(
        self, model: torch.nn.Module, batch: Any, batch_idx: int, metrics: Optional[Any] = None
    ) -> BATCH_OUTPUT:
        raise NotImplementedError

    def _compute_num_training_steps(self, train_loader: DataLoader) -> int:
        # FIXME: when accumulate batches is added
        return len(train_loader)

    def _is_done(self, batch_idx: int, dry_run: Optional[bool], limit_batches: Optional[int]) -> bool:
        if dry_run is not None and dry_run is True:
            return True

        if limit_batches is not None and limit_batches <= batch_idx:
            return True

        return False
