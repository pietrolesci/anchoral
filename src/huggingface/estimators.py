from typing import Callable, Dict, Optional, Union

import torch
from torchmetrics import MetricCollection
from torchmetrics.classification import Accuracy, F1Score

from src.enums import InputKeys, OutputKeys, RunningStage, SpecialKeys
from src.estimator import Estimator
from src.query_strategies.base import UncertaintyBasedStrategy
from src.types import BATCH_OUTPUT, EPOCH_OUTPUT, EVAL_BATCH_OUTPUT, POOL_BATCH_OUTPUT


class SequenceClassificationMixin:
    def step(
        self,
        loss_fn: Optional[Union[torch.nn.Module, Callable]],
        model: torch.nn.Module,
        batch: Dict,
        batch_idx: int,
        metrics: MetricCollection,
    ) -> BATCH_OUTPUT:
        unique_ids = batch.pop(InputKeys.ON_CPU)[SpecialKeys.ID]

        # forward pass
        out = model(**batch)

        # compute loss
        loss = loss_fn(out.logits, batch[InputKeys.TARGET])

        # compute metrics
        preds = out.logits.argmax(-1)
        out_metrics = metrics(preds, batch[InputKeys.TARGET])

        return {
            OutputKeys.LOSS: loss,
            OutputKeys.LOGITS: out.logits,
            OutputKeys.METRICS: out_metrics,
            SpecialKeys.ID: unique_ids,
        }

    def train_step(
        self,
        loss_fn: Optional[Union[torch.nn.Module, Callable]],
        model: torch.nn.Module,
        batch: Dict,
        batch_idx: int,
        metrics: MetricCollection,
    ) -> BATCH_OUTPUT:
        return self.step(loss_fn, model, batch, batch_idx, metrics)

    def validation_step(
        self,
        loss_fn: Optional[Union[torch.nn.Module, Callable]],
        model: torch.nn.Module,
        batch: Dict,
        batch_idx: int,
        metrics: MetricCollection,
    ) -> EVAL_BATCH_OUTPUT:
        return self.step(loss_fn, model, batch, batch_idx, metrics)

    def test_step(
        self,
        loss_fn: Optional[Union[torch.nn.Module, Callable]],
        model: torch.nn.Module,
        batch: Dict,
        batch_idx: int,
        metrics: MetricCollection,
    ) -> EVAL_BATCH_OUTPUT:
        return self.step(loss_fn, model, batch, batch_idx, metrics)

    def configure_metrics(self, stage: Optional[RunningStage] = None) -> Optional[MetricCollection]:
        # you are in charge of moving it to the correct device
        if stage != RunningStage.POOL:
            return MetricCollection(
                {
                    "accuracy": Accuracy("multiclass", num_classes=self.model.num_labels),
                    "f1_macro": F1Score("multiclass", num_classes=self.model.num_labels, average="macro"),
                }
            ).to(self.device)

    def log(self, output: BATCH_OUTPUT, batch_idx: int, stage: RunningStage) -> None:
        self.fabric.log(OutputKeys.LOSS, output[OutputKeys.LOSS], step=batch_idx)
        self.fabric.log_dict(output[OutputKeys.METRICS], step=batch_idx)

    def epoch_end(self, output: EPOCH_OUTPUT, metrics: MetricCollection) -> EPOCH_OUTPUT:
        # delete data to save space
        return

    def train_epoch_end(self, output: EPOCH_OUTPUT, metrics: MetricCollection) -> EPOCH_OUTPUT:
        return self.epoch_end(output, metrics)

    def validation_epoch_end(self, output: EPOCH_OUTPUT, metrics: MetricCollection) -> EPOCH_OUTPUT:
        return self.epoch_end(output, metrics)

    def test_epoch_end(self, output: EPOCH_OUTPUT, metrics: MetricCollection) -> EPOCH_OUTPUT:
        return self.epoch_end(output, metrics)


class EstimatorForSequenceClassification(SequenceClassificationMixin, Estimator):
    pass


class UncertaintyBasedStrategyForSequenceClassification(SequenceClassificationMixin, UncertaintyBasedStrategy):
    def pool_step(
        self, model: torch.nn.Module, batch: Dict, batch_idx: int, metrics: MetricCollection
    ) -> POOL_BATCH_OUTPUT:
        logits = model(**batch).logits
        return self.score_fn(logits)
