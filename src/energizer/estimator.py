import inspect
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Union

import torch
from lightning.fabric import Fabric
from lightning.fabric.accelerators.accelerator import Accelerator
from lightning.fabric.connector import _PRECISION_INPUT
from lightning.fabric.loggers import Logger
from lightning.fabric.wrappers import _FabricDataLoader, _FabricModule, _FabricOptimizer
from lightning.pytorch.core.mixins.hparams_mixin import HyperparametersMixin
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

from src.energizer.enums import OutputKeys, RunningStage
from src.energizer.progress_trackers import ProgressTracker
from src.energizer.registries import LOSS_FUNCTIONS_REGISTRY, OPTIMIZER_REGISTRY, SCHEDULER_REGISTRY
from src.energizer.types import BATCH_OUTPUT, EPOCH_OUTPUT, METRIC
from src.energizer.utilities import init_deterministic, move_to_cpu
from src.energizer.utilities.model_summary import summarize


@dataclass
class FitEpochOutput:
    """Simple container for train and validation outputs for each epoch of fitting.

    This deserves a separate container bacause during fit we obtain EpochOutput's
    from both train and, possibly, validation.
    """

    train: EPOCH_OUTPUT = None
    validation: EPOCH_OUTPUT = None


class Estimator(HyperparametersMixin):
    _hparams_ignore: List[str] = ["model", "loggers", "callbacks"]
    _loss_fn: Optional[Union[torch.nn.Module, Callable]] = None
    _progress_tracker: ProgressTracker = None

    def __init__(
        self,
        model: torch.nn.Module,
        accelerator: Optional[Union[str, Accelerator]] = None,
        precision: _PRECISION_INPUT = 32,
        callbacks: Optional[Union[List[Any], Any]] = None,
        loggers: Optional[Union[Logger, List[Logger]]] = None,
        deterministic: bool = True,
    ) -> None:
        super().__init__()
        self.fabric = Fabric(
            accelerator=accelerator,
            precision=precision,
            callbacks=callbacks,
            loggers=loggers,
        )
        self.model = model
        init_deterministic(deterministic)
        self.save_hyperparameters(ignore=self._hparams_ignore)

    @property
    def device(self) -> torch.device:
        return self.fabric.device

    @property
    def progress_tracker(self) -> ProgressTracker:
        if self._progress_tracker is None:
            self._progress_tracker = ProgressTracker()
        return self._progress_tracker

    @property
    def model_summary(self) -> str:
        return summarize(self)

    """
    Main methods
    """

    def fit(
        self,
        train_loader: DataLoader,
        validation_loader: Optional[DataLoader] = None,
        max_epochs: Optional[int] = 3,
        min_steps: Optional[int] = None,
        loss_fn: Optional[Union[str, torch.nn.Module, Callable]] = None,
        loss_fn_kwargs: Optional[Dict] = None,
        learning_rate: float = 0.001,
        optimizer: str = "adamw",
        optimizer_kwargs: Optional[Dict] = None,
        scheduler: Optional[str] = None,
        scheduler_kwargs: Optional[Dict] = None,
        **kwargs,
    ) -> List[FitEpochOutput]:
        """Runs full training and validation.
        Kwargs: log_interval: int, limit_train_batches: int, limit_validation_batches: int
        """

        # configure progress tracking
        self.progress_tracker.initialize_fit_progress(
            max_epochs,
            min_steps,
            train_loader,
            validation_loader,
            has_validation=validation_loader is not None,
            **kwargs,
        )

        # configure dataloaders
        train_loader = self.configure_dataloader(train_loader)
        validation_loader = self.configure_dataloader(validation_loader)

        # configure optimizer and scheduler
        optimizer = self.configure_optimizer(optimizer, learning_rate, optimizer_kwargs)
        scheduler = self.configure_scheduler(scheduler, optimizer, scheduler_kwargs)

        # configure loss
        loss_fn = self.configure_loss_fn(loss_fn, loss_fn_kwargs, RunningStage.TRAIN)

        # setup model and optimizer with fabric
        model, optimizer = self.fabric.setup(self.model, optimizer)

        # call hook
        self.fabric.call("on_fit_start", estimator=self, model=model)

        output = []
        while not self.progress_tracker.is_fit_done():
            out = self.fit_epoch_loop(
                model=model,
                train_loader=train_loader,
                validation_loader=validation_loader,
                loss_fn=loss_fn,
                optimizer=optimizer,
                scheduler=scheduler,
            )

            output.append(out)

            # update progress
            self.progress_tracker.increment_fit_progress()

        # call hook
        self.fabric.call("on_fit_end", estimator=self, model=model, output=output)

        self.progress_tracker.finalize_fit_progress()

        return output

    def validate(
        self,
        validation_loader: DataLoader,
        loss_fn: Optional[Union[torch.nn.Module, Callable]] = None,
        loss_fn_kwargs: Optional[Dict] = None,
        **kwargs,
    ) -> EPOCH_OUTPUT:
        """Runs validation.

        Kwargs that can be passed:, log_interval: int, limit_batches: int, progress_bar: bool
        """
        return self._evaluate(validation_loader, loss_fn, loss_fn_kwargs, RunningStage.VALIDATION, **kwargs)

    def test(
        self,
        test_loader: DataLoader,
        loss_fn: Optional[Union[torch.nn.Module, Callable]] = None,
        loss_fn_kwargs: Optional[Dict] = None,
        **kwargs,
    ) -> EPOCH_OUTPUT:
        """Runs testing.

        Kwargs that can be passed:, log_interval: int, limit_batches: int, progress_bar: bool
        """
        return self._evaluate(test_loader, loss_fn, loss_fn_kwargs, RunningStage.TEST, **kwargs)

    def _evaluate(
        self,
        loader: DataLoader,
        loss_fn: Optional[Union[torch.nn.Module, Callable]],
        loss_fn_kwargs: Optional[Dict],
        stage: RunningStage,
        **kwargs,
    ) -> EPOCH_OUTPUT:
        """This method is useful because validation can run in fit when model is already setup."""

        # configure dataloader
        loader = self.configure_dataloader(loader)

        # define loss function
        loss_fn = self.configure_loss_fn(loss_fn, loss_fn_kwargs, RunningStage.VALIDATION)

        # configure model
        model = self.fabric.setup(self.model)

        return self.eval_loop(loss_fn, model, loader, stage, **kwargs)

    """
    Loops
    """

    def fit_epoch_loop(
        self,
        model: _FabricModule,
        train_loader: _FabricDataLoader,
        validation_loader: Optional[_FabricDataLoader],
        loss_fn: Optional[Union[torch.nn.Module, Callable]],
        optimizer: _FabricOptimizer,
        scheduler: Optional[str],
    ) -> FitEpochOutput:
        """Runs a training epoch."""

        # configure progress tracking
        self.progress_tracker.initialize_epoch_progress(RunningStage.TRAIN)

        # define metrics
        metrics = self.configure_metrics(RunningStage.TRAIN)

        # train mode
        model.train()

        # call hook
        self.fabric.call("on_train_epoch_start", estimator=self, model=model, optimizer=optimizer)

        train_out, validation_out = [], []
        for batch_idx, batch in enumerate(train_loader):

            # check stopping condition
            if self.progress_tracker.is_epoch_done():
                break

            # validation loop
            if self.progress_tracker.should_validate():
                out = self.eval_loop(loss_fn, model, validation_loader, RunningStage.VALIDATION)
                if out is not None:
                    validation_out.append(out)
                self.progress_tracker.continue_epoch_progress(RunningStage.TRAIN)  # continue training tracking

            # put batch on correct device
            batch = self.transfer_to_device(batch)

            # call hook
            self.fabric.call(
                "on_train_batch_start",
                estimator=self,
                model=model,
                optimizer=optimizer,
                batch=batch,
                batch_idx=batch_idx,
            )

            # run model on batch
            batch_out = self.training_step(loss_fn, model, batch, batch_idx, optimizer, scheduler, metrics)

            # call hook
            self.fabric.call(
                "on_train_batch_end",
                estimator=self,
                model=model,
                output=batch_out,
                batch=batch,
                batch_idx=batch_idx,
            )

            # record output
            train_out.append(move_to_cpu(batch_out))

            # update progress tracker
            self.progress_tracker.increment_epoch_progress()

        # method to possibly aggregate
        train_out = self.train_epoch_end(train_out, metrics)

        # call hook
        self.fabric.call(
            "on_train_epoch_end",
            estimator=self,
            model=model,
            output=train_out,
            metrics=metrics,
        )

        self.progress_tracker.finalize_epoch_progress()

        # validation loop
        if self.progress_tracker.should_validate():
            out = self.eval_loop(loss_fn, model, validation_loader, RunningStage.VALIDATION)
            if out is not None:
                validation_out.append(out)
            self.progress_tracker.continue_epoch_progress(RunningStage.TRAIN)  # continue training tracking

        return FitEpochOutput(train=train_out, validation=validation_out)

    def eval_loop(
        self,
        loss_fn: Optional[Union[torch.nn.Module, Callable]],
        model: _FabricModule,
        loader: _FabricDataLoader,
        stage: RunningStage,
        **kwargs,
    ) -> EPOCH_OUTPUT:
        """Runs over an entire evaluation dataloader."""

        # configure progress tracking
        self.progress_tracker.initialize_epoch_progress(stage, loader=loader, **kwargs)

        # configure metrics
        metrics = self.configure_metrics(stage)

        # eval mode
        is_fitting = model.training
        model.eval()

        # call hook
        self.fabric.call(f"on_{stage}_epoch_start", estimator=self, model=model)

        output = []
        with torch.inference_mode():
            for batch_idx, batch in enumerate(loader):
                # check stopping condition
                if self.progress_tracker.is_epoch_done():
                    break

                # put batch on correct device
                batch = self.transfer_to_device(batch)

                # call hook
                self.fabric.call(
                    f"on_{stage}_batch_start", estimator=self, model=model, batch=batch, batch_idx=batch_idx
                )

                # run model on batch
                batch_out = self.evaluation_step(loss_fn, model, batch, batch_idx, metrics, stage)

                # call hook
                self.fabric.call(
                    f"on_{stage}_batch_end",
                    estimator=self,
                    model=model,
                    output=batch_out,
                    batch=batch,
                    batch_idx=batch_idx,
                )

                # record output
                if batch_out is not None:
                    output.append(move_to_cpu(batch_out))

                # update progress tracker
                self.progress_tracker.increment_epoch_progress()

        # method to possibly aggregate
        output = getattr(self, f"{stage}_epoch_end")(output, metrics)

        # call hook
        self.fabric.call(f"on_{stage}_epoch_end", estimator=self, model=model, output=output, metrics=metrics)

        # resets model training status
        model.train(is_fitting)

        self.progress_tracker.finalize_epoch_progress()

        return output

    def training_step(
        self,
        loss_fn: Optional[Union[torch.nn.Module, Callable]],
        model: _FabricModule,
        batch: Any,
        batch_idx: int,
        optimizer: _FabricOptimizer,
        scheduler: _LRScheduler,
        metrics: Optional[METRIC],
    ) -> BATCH_OUTPUT:
        """Runs over a single batch of data."""

        # zero_grad
        optimizer.zero_grad()

        # compute loss
        output = self.train_step(loss_fn, model, batch, batch_idx, metrics)
        loss = output if isinstance(output, torch.Tensor) else output[OutputKeys.LOSS]

        # compute gradients
        self.fabric.backward(loss)  # instead of loss.backward()

        # update parameters
        optimizer.step()

        # update scheduler
        if scheduler is not None:
            scheduler.step()

        # update progress_tracker
        self.progress_tracker.increment_step_progress()

        return output

    def evaluation_step(
        self,
        loss_fn: Optional[Union[torch.nn.Module, Callable]],
        model: _FabricModule,
        batch: Any,
        batch_idx: int,
        metrics: Optional[METRIC],
        stage: RunningStage,
    ) -> Optional[BATCH_OUTPUT]:
        """Runs over a single batch of data."""
        # this might seems redundant but it's useful for active learning to hook in
        return getattr(self, f"{stage}_step")(loss_fn, model, batch, batch_idx, metrics)

    """
    Methods
    """

    def transfer_to_device(self, batch: Any) -> Any:
        return self.fabric.to_device(batch)

    def configure_optimizer(
        self, optimizer: str, learning_rate: float, optimizer_kwargs: Optional[Dict] = None
    ) -> Optimizer:
        assert optimizer is not None, ValueError("You must provide an optimizer.")

        optimizer_fn = OPTIMIZER_REGISTRY[optimizer]
        optimizer_kwargs = optimizer_kwargs or {}

        # weight decay
        no_decay, weight_decay = optimizer_kwargs.get("no_decay", None), optimizer_kwargs.get("weight_decay", None)
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
        optimizer = optimizer_fn(params, lr=learning_rate, **optimizer_kwargs)

        return optimizer

    def configure_scheduler(
        self,
        scheduler: str,
        optimizer: Optimizer,
        scheduler_kwargs: Optional[Dict] = None,
    ) -> Optional[_LRScheduler]:
        if scheduler is None:
            return

        scheduler_fn = SCHEDULER_REGISTRY[scheduler]

        # collect scheduler kwargs
        params = list(inspect.signature(scheduler_fn).parameters.keys())
        scheduler_kwargs = scheduler_kwargs or {}
        num_train_steps = self.progress_tracker.fit_tracker.train_tracker.max
        num_warmup_steps = scheduler_kwargs.get("num_warmup_steps", None)
        if num_warmup_steps is not None and isinstance(num_warmup_steps, float):
            num_warmup_steps *= num_train_steps
        if "num_train_steps" in params:
            scheduler_kwargs["num_train_steps"] = num_train_steps
        if "num_warmup_steps" in params:
            scheduler_kwargs["num_warmup_steps"] = num_warmup_steps

        scheduler = scheduler_fn(optimizer, **scheduler_kwargs)

        return scheduler

    def configure_loss_fn(
        self,
        loss_fn: Optional[Union[str, torch.nn.Module, Callable]],
        loss_fn_kwargs: Optional[Dict],
        stage: RunningStage,
    ) -> Optional[Union[torch.nn.Module, Callable]]:
        if loss_fn is None:
            # if we have already trained, this will return the same loss_fn used during training
            return self._loss_fn

        loss_fn_kwargs = loss_fn_kwargs or {}

        # get class or function from registry
        if isinstance(loss_fn, str):
            loss_fn = LOSS_FUNCTIONS_REGISTRY[loss_fn]

        if isinstance(loss_fn, Callable):
            # functional
            loss_fn = partial(loss_fn, **loss_fn_kwargs)
        else:
            # module
            loss_fn = loss_fn(**loss_fn_kwargs)

        if stage == RunningStage.TRAIN:
            self._loss_fn = loss_fn

        return loss_fn

    def configure_dataloader(self, loader: Optional[DataLoader]) -> Optional[_FabricDataLoader]:
        """Does not move the dataloader to the device."""
        if loader is None:
            return

        return self.fabric.setup_dataloaders(loader, replace_sampler=False, move_to_device=False)

    """
    Hooks
    """

    def configure_metrics(self, stage: Optional[RunningStage] = None) -> Optional[METRIC]:
        pass

    def forward_pass(self, model: _FabricModule, batch: Any) -> Any:
        return model(batch)

    def compute_loss(
        self,
        loss_fn: Optional[Union[torch.nn.Module, Callable]],
        model: _FabricModule,
        batch: Any,
    ) -> torch.Tensor:
        preds = self.forward_pass(model, batch)
        return loss_fn(preds, batch)

    def train_step(
        self,
        loss_fn: Optional[Union[torch.nn.Module, Callable]],
        model: _FabricModule,
        batch: Any,
        batch_idx: int,
        metrics: Optional[METRIC] = None,
    ) -> BATCH_OUTPUT:
        return self.compute_loss(loss_fn, model, batch)

    def validation_step(
        self,
        loss_fn: Optional[Union[torch.nn.Module, Callable]],
        model: _FabricModule,
        batch: Any,
        batch_idx: int,
        metrics: Optional[METRIC] = None,
    ) -> Optional[BATCH_OUTPUT]:
        return self.compute_loss(loss_fn, model, batch)

    def test_step(
        self,
        loss_fn: Optional[Union[torch.nn.Module, Callable]],
        model: _FabricModule,
        batch: Any,
        batch_idx: int,
        metrics: Optional[METRIC] = None,
    ) -> Optional[BATCH_OUTPUT]:
        return self.compute_loss(loss_fn, model, batch)

    def train_epoch_end(self, output: List[BATCH_OUTPUT], metrics: Optional[METRIC]) -> EPOCH_OUTPUT:
        return output

    def validation_epoch_end(self, output: List, metrics: Optional[METRIC]) -> EPOCH_OUTPUT:
        return output

    def test_epoch_end(self, output: List, metrics: Optional[METRIC]) -> EPOCH_OUTPUT:
        return output

    def log(self, name: str, value: Any, step: int) -> None:
        """Automatically moves to cpu and then logs value."""
        if self.progress_tracker.should_log():
            self.fabric.log(name, move_to_cpu(value), step)

    def log_dict(self, value_dict: Mapping[str, Any], step: int) -> None:
        """Automatically moves to cpu and then logs mapping of values."""
        if self.progress_tracker.should_log():
            self.fabric.log_dict(value_dict, step)

    def save_state_dict(self, cache_dir: Union[str, Path], name: str = "state_dict.pt") -> None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.fabric.save(self.model.state_dict(), cache_dir / name)

    def load_state_dict(self, cache_dir: Union[str, Path], name: str = "state_dict.pt") -> None:
        cache_dir = Path(cache_dir)
        self.model.load_state_dict(self.fabric.load(cache_dir / name))
