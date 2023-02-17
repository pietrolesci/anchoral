from typing import Callable, Dict, MutableMapping, Optional, Union

import numpy as np
import torch
from lightning.fabric.wrappers import _FabricModule
from lightning.pytorch.utilities.parsing import AttributeDict
from torchmetrics import MetricCollection
from torchmetrics.classification import Accuracy, F1Score
from transformers import AutoModelForSequenceClassification

from src.active_learning.strategies import RandomStrategy, UncertaintyBasedStrategy
from src.enums import InputKeys, OutputKeys, RunningStage, SpecialKeys
from src.estimator import Estimator
from src.types import BATCH_OUTPUT, EPOCH_OUTPUT, EVAL_BATCH_OUTPUT, POOL_BATCH_OUTPUT
from src.utilities import move_to_cpu


class SequenceClassificationMixin:

    """
    Method forwarding
    """

    def train_step(
        self,
        loss_fn: Optional[Union[torch.nn.Module, Callable]],
        model: _FabricModule,
        batch: Dict,
        batch_idx: int,
        metrics: MetricCollection,
    ) -> BATCH_OUTPUT:
        return self.step(loss_fn, model, batch, batch_idx, metrics)

    def validation_step(
        self,
        loss_fn: Optional[Union[torch.nn.Module, Callable]],
        model: _FabricModule,
        batch: Dict,
        batch_idx: int,
        metrics: MetricCollection,
    ) -> EVAL_BATCH_OUTPUT:
        return self.step(loss_fn, model, batch, batch_idx, metrics)

    def test_step(
        self,
        loss_fn: Optional[Union[torch.nn.Module, Callable]],
        model: _FabricModule,
        batch: Dict,
        batch_idx: int,
        metrics: MetricCollection,
    ) -> EVAL_BATCH_OUTPUT:
        return self.step(loss_fn, model, batch, batch_idx, metrics)

    def train_epoch_end(self, output: EPOCH_OUTPUT, metrics: MetricCollection) -> EPOCH_OUTPUT:
        return self.epoch_end(output, metrics, RunningStage.TRAIN)

    def validation_epoch_end(self, output: EPOCH_OUTPUT, metrics: MetricCollection) -> EPOCH_OUTPUT:
        return self.epoch_end(output, metrics, RunningStage.VALIDATION)

    def test_epoch_end(self, output: EPOCH_OUTPUT, metrics: MetricCollection) -> EPOCH_OUTPUT:
        return self.epoch_end(output, metrics, RunningStage.TEST)

    def train_step_end(self, output: BATCH_OUTPUT, batch: Dict, batch_idx: int, log_interval: int) -> BATCH_OUTPUT:
        return self.step_end(output, batch, batch_idx, log_interval, RunningStage.TRAIN)

    def validation_step_end(self, output: BATCH_OUTPUT, batch: Dict, batch_idx: int, log_interval: int) -> BATCH_OUTPUT:
        return self.step_end(output, batch, batch_idx, log_interval, RunningStage.VALIDATION)

    def test_step_end(self, output: BATCH_OUTPUT, batch: Dict, batch_idx: int, log_interval: int) -> BATCH_OUTPUT:
        return self.step_end(output, batch, batch_idx, log_interval, RunningStage.TEST)

    """
    Changes
    """

    @property
    def hparams(self) -> Union[AttributeDict, MutableMapping]:
        hparams = super().hparams
        hparams["name_or_path"] = self.model.name_or_path
        return hparams

    def configure_loss_fn(
        self,
        loss_fn: Optional[Union[str, torch.nn.Module, Callable]],
        loss_fn_kwargs: Optional[Dict],
        stage: RunningStage,
    ) -> Optional[Union[torch.nn.Module, Callable]]:
        if loss_fn_kwargs is not None and "weight" in loss_fn_kwargs:
            loss_fn_kwargs["weight"] = torch.tensor(loss_fn_kwargs["weight"], dtype=torch.float32, device=self.device)

        return super().configure_loss_fn(loss_fn, loss_fn_kwargs, stage)

    def configure_metrics(self, stage: Optional[RunningStage] = None) -> Optional[MetricCollection]:
        if stage == RunningStage.POOL:
            return
        # you are in charge of moving it to the correct device
        return MetricCollection(
            {
                "accuracy": Accuracy("multiclass", num_classes=self.model.num_labels),
                "f1_macro": F1Score("multiclass", num_classes=self.model.num_labels, average="macro"),
            }
        ).to(self.device)

    def step(
        self,
        loss_fn: Optional[Union[torch.nn.Module, Callable]],
        model: AutoModelForSequenceClassification,
        batch: Dict,
        batch_idx: int,
        metrics: MetricCollection,
    ) -> BATCH_OUTPUT:
        unique_ids = batch.pop(InputKeys.ON_CPU)[SpecialKeys.ID]

        # forward pass
        out = model(**batch)

        # compute weighted loss if provided otherwise
        # used the unweighted loss automaticaly computed by transformers
        loss = loss_fn(out.logits, batch[InputKeys.TARGET]) if loss_fn is not None else out.loss

        # compute metrics
        out_metrics = metrics(out.logits, batch[InputKeys.TARGET])

        return {
            OutputKeys.LOSS: loss,
            OutputKeys.LOGITS: out.logits,
            OutputKeys.METRICS: out_metrics,
            SpecialKeys.ID: unique_ids,
        }

    def step_end(
        self, output: BATCH_OUTPUT, batch: Dict, batch_idx: int, log_interval: int, stage: RunningStage
    ) -> BATCH_OUTPUT:
        # NOTE: only log at the batch level for the training loop
        if stage != RunningStage.TRAIN:
            return output

        # control logging interval
        if (batch_idx == 0) or ((batch_idx + 1) % log_interval == 0):
            # NOTE: output is still on device
            logs = {OutputKeys.LOSS: output[OutputKeys.LOSS], **output[OutputKeys.METRICS]}

            # rename and move to cpu
            logs = move_to_cpu({f"{stage}/{k}": v for k, v in logs.items()})

            # log
            self.fabric.log_dict(logs, step=self.counter.get_batch_step(stage, batch_idx))

        return output

    def epoch_end(self, output: EPOCH_OUTPUT, metrics: MetricCollection, stage: RunningStage) -> EPOCH_OUTPUT:
        # during training do not collect nor aggregate any metric
        if stage == RunningStage.TRAIN:
            return

        # NOTE: the metric object and is on device but the output is already on cpu
        logs = {
            **move_to_cpu(metrics.compute()),
            "avg_loss": round(np.mean([out[OutputKeys.LOSS] for out in output]), 6),
        }

        # rename
        logs = {f"{stage}_end/{k}": v for k, v in logs.items()}

        # log
        self.fabric.log_dict(logs, step=self.counter.get_epoch_step(stage))

        return logs


class EstimatorForSequenceClassification(SequenceClassificationMixin, Estimator):
    ...


class UncertaintyBasedStrategyForSequenceClassification(SequenceClassificationMixin, UncertaintyBasedStrategy):
    def pool_step(
        self,
        model: _FabricModule,
        batch: Dict,
        batch_idx: int,
        metrics: Optional[MetricCollection] = None,
    ) -> POOL_BATCH_OUTPUT:
        _ = batch.pop(InputKeys.ON_CPU)  # this is already handled in the `eval_batch_loop`

        logits = model(**batch).logits
        scores = self.score_fn(logits)

        return {OutputKeys.SCORES: scores, OutputKeys.LOGITS: logits}


class RandomStrategyForSequenceClassification(SequenceClassificationMixin, RandomStrategy):
    pass
