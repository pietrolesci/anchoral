import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Union

import torch
from lightning.fabric import Fabric
from lightning.fabric.accelerators.accelerator import Accelerator
from lightning.fabric.connector import _PLUGIN_INPUT, _PRECISION_INPUT
from lightning.fabric.loggers import Logger
from lightning.fabric.strategies import Strategy
from lightning.fabric.wrappers import _FabricDataLoader, _FabricModule, _FabricOptimizer
from lightning.pytorch.core.mixins.hparams_mixin import HyperparametersMixin
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

from src.energizer.enums import OutputKeys, RunningStage
from src.energizer.progress_trackers import ProgressTracker
from src.energizer.registries import OPTIMIZER_REGISTRY, SCHEDULER_REGISTRY
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
    _progress_tracker: ProgressTracker = None

    def __init__(
        self,
        model: torch.nn.Module,
        accelerator: Optional[Union[str, Accelerator]] = None,
        precision: _PRECISION_INPUT = 32,
        callbacks: Optional[Union[List[Any], Any]] = None,
        loggers: Optional[Union[Logger, List[Logger]]] = None,
        deterministic: bool = True,
        # strategy: Optional[Union[str, Strategy]] = None,
        # devices: Optional[Union[List[int], str, int]] = None,
        # num_nodes: int = 1,
        # plugins: Optional[Union[_PLUGIN_INPUT, List[_PLUGIN_INPUT]]] = None,
    ) -> None:
        super().__init__()
        self.fabric = Fabric(
            accelerator=accelerator,
            precision=precision,
            callbacks=callbacks,
            loggers=loggers,
            # strategy=strategy,
            # devices=devices,
            # num_nodes=num_nodes,
            # plugins=plugins,
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
        learning_rate: float = 0.001,
        optimizer: str = "adamw",
        optimizer_kwargs: Optional[Dict] = None,
        scheduler: Optional[str] = None,
        scheduler_kwargs: Optional[Dict] = None,
        max_epochs: Optional[int] = 3,
        max_steps: Optional[int] = None,
        min_epochs: Optional[int] = 3,
        min_steps: Optional[int] = None,
        validation_frequency: Optional[float] = None,
        limit_train_batches: Optional[int] = None,
        limit_validation_batches: Optional[int] = None,
        progress_bar: Optional[bool] = True,
        log_interval: Optional[int] = 1,
    ) -> List[FitEpochOutput]:

        # configure progress tracking
        self.progress_tracker.initialize_fit_progress(
            progress_bar=progress_bar,
            log_interval=log_interval,
            num_train_batches=len(train_loader),
            max_epochs=max_epochs,
            max_steps=max_steps,
            min_epochs=min_epochs,
            min_steps=min_steps,
            has_validation=validation_loader is not None,
        )

        # configure dataloaders
        train_loader = self.configure_dataloader(train_loader)
        validation_loader = self.configure_dataloader(validation_loader)

        # configure optimizer and scheduler
        optimizer = self.configure_optimizer(optimizer, learning_rate, optimizer_kwargs)
        scheduler = self.configure_scheduler(scheduler, optimizer, scheduler_kwargs)

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
                optimizer=optimizer,
                scheduler=scheduler,
                limit_train_batches=limit_train_batches,
                limit_validation_batches=limit_validation_batches,
                validation_frequency=validation_frequency,
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
        limit_batches: Optional[int] = None,
        progress_bar: Optional[bool] = True,
        log_interval: Optional[int] = 1,
    ) -> EPOCH_OUTPUT:
        """Runs validation.

        Kwargs that can be passed:, log_interval: int, limit_batches: int, progress_bar: bool
        """
        loader = self.configure_dataloader(validation_loader)
        model = self.fabric.setup(self.model)
        return self.eval_loop(
            model,
            loader,
            RunningStage.VALIDATION,
            limit_batches=limit_batches,
            progress_bar=progress_bar,
            log_interval=log_interval,
        )

    def test(
        self,
        test_loader: DataLoader,
        limit_batches: Optional[int] = None,
        progress_bar: Optional[bool] = True,
        log_interval: Optional[int] = 1,
    ) -> EPOCH_OUTPUT:
        """Runs testing.

        Kwargs that can be passed:, log_interval: int, limit_batches: int, progress_bar: bool
        """
        loader = self.configure_dataloader(test_loader)
        model = self.fabric.setup(self.model)
        return self.eval_loop(
            model,
            loader,
            RunningStage.TEST,
            limit_batches=limit_batches,
            progress_bar=progress_bar,
            log_interval=log_interval,
        )

    """
    Loops
    """

    def fit_epoch_loop(
        self,
        model: _FabricModule,
        train_loader: _FabricDataLoader,
        validation_loader: Optional[_FabricDataLoader],
        optimizer: _FabricOptimizer,
        scheduler: Optional[str],
        validation_frequency: Optional[float] = None,
        limit_train_batches: Optional[int] = None,
        limit_validation_batches: Optional[int] = None,
    ) -> FitEpochOutput:
        """Runs a training epoch."""

        # start progress tracking
        self.progress_tracker.initialize_epoch_progress(
            train_loader,
            RunningStage.TRAIN,
            validation_frequency=validation_frequency,
            limit_train_batches=limit_train_batches,
        )

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

            # validate mid-epoch
            if self.progress_tracker.should_validate():
                out = self.eval_loop(
                    model, validation_loader, RunningStage.VALIDATION, limit_validation_batches=limit_validation_batches
                )
                if out is not None:
                    validation_out.append(out)
                # NOTE: continue training tracking -> re-attach train_tracker since it gets changed by `eval_loop`
                self.progress_tracker.current_stage = RunningStage.TRAIN

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
            batch_out = self.training_step(model, batch, batch_idx, optimizer, scheduler, metrics)

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

        # validate after training
        if self.progress_tracker.should_validate():
            out = self.eval_loop(
                model, validation_loader, RunningStage.VALIDATION, limit_validation_batches=limit_validation_batches
            )
            if out is not None:
                validation_out.append(out)

        # end progress tracking
        self.progress_tracker.finalize_epoch_progress()

        return FitEpochOutput(train=train_out, validation=validation_out)

    def eval_loop(
        self,
        model: _FabricModule,
        loader: _FabricDataLoader,
        stage: RunningStage,
        **kwargs,
    ) -> EPOCH_OUTPUT:
        """Runs over an entire evaluation dataloader."""

        # start progress tracking
        self.progress_tracker.initialize_epoch_progress(loader, stage, **kwargs)

        # configure metrics
        metrics = self.configure_metrics(stage)

        # eval mode
        is_training = model.training
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
                batch_out = self.evaluation_step(
                    model=model, batch=batch, batch_idx=batch_idx, metrics=metrics, stage=stage
                )

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
        model.train(is_training)

        # end progress tracking
        self.progress_tracker.finalize_epoch_progress()

        return output

    def training_step(
        self,
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
        output = self.train_step(model, batch, batch_idx, metrics)
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
        model: _FabricModule,
        batch: Any,
        batch_idx: int,
        metrics: Optional[METRIC],
        stage: RunningStage,
    ) -> Optional[BATCH_OUTPUT]:
        """Runs over a single batch of data."""
        # this might seems redundant but it's useful for active learning to hook in
        return getattr(self, f"{stage}_step")(model, batch, batch_idx, metrics)

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

    def configure_dataloader(self, loader: Optional[DataLoader]) -> Optional[_FabricDataLoader]:
        """Does not move the dataloader to the device."""
        if loader is None:
            return

        return self.fabric.setup_dataloaders(loader, replace_sampler=False, move_to_device=False)

    def configure_metrics(self, stage: Optional[RunningStage] = None) -> Optional[METRIC]:
        ...

    def train_step(
        self,
        model: _FabricModule,
        batch: Any,
        batch_idx: int,
        metrics: Optional[METRIC] = None,
    ) -> BATCH_OUTPUT:
        raise NotImplementedError

    def validation_step(
        self,
        model: _FabricModule,
        batch: Any,
        batch_idx: int,
        metrics: Optional[METRIC] = None,
    ) -> Optional[BATCH_OUTPUT]:
        raise NotImplementedError

    def test_step(
        self,
        model: _FabricModule,
        batch: Any,
        batch_idx: int,
        metrics: Optional[METRIC] = None,
    ) -> Optional[BATCH_OUTPUT]:
        raise NotImplementedError

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
