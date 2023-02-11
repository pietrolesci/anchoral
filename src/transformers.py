from typing import Dict, Optional, Union

import torch
from torchmetrics import MetricCollection
from torchmetrics.classification import Accuracy, F1Score

from src.containers import EpochOutput
from src.enums import InputKeys, OutputKeys, RunningStage, SpecialKeys
from src.estimator import Estimator
from src.query_strategies.base import UncertaintyBasedStrategy
from src.types import BATCH_OUTPUT, EVAL_BATCH_OUTPUT, POOL_BATCH_OUTPUT


class SequenceClassificationMixin:
    def step(self, model: torch.nn.Module, batch: Dict, batch_idx: int, metrics: MetricCollection) -> BATCH_OUTPUT:
        unique_ids = batch.pop(InputKeys.ON_CPU)[SpecialKeys.ID]

        out = model(**batch)
        preds = out.logits.argmax(-1)
        out_metrics = metrics(preds, batch[InputKeys.TARGET])
        return {
            OutputKeys.LOSS: out.loss,
            OutputKeys.LOGITS: out.logits,
            OutputKeys.METRICS: out_metrics,
            SpecialKeys.ID: unique_ids,
        }

    def training_step(
        self, model: torch.nn.Module, batch: Dict, batch_idx: int, metrics: MetricCollection
    ) -> BATCH_OUTPUT:
        return self.step(model, batch, batch_idx, metrics)

    def validation_step(
        self, model: torch.nn.Module, batch: Dict, batch_idx: int, metrics: MetricCollection
    ) -> EVAL_BATCH_OUTPUT:
        return self.step(model, batch, batch_idx, metrics)

    def test_step(
        self, model: torch.nn.Module, batch: Dict, batch_idx: int, metrics: MetricCollection
    ) -> EVAL_BATCH_OUTPUT:
        return self.step(model, batch, batch_idx, metrics)

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
        if "loss" in output:
            self.fabric.log(OutputKeys.LOSS, output[OutputKeys.LOSS], step=batch_idx)
        if "metrics" in output:
            self.fabric.log_dict(output[OutputKeys.METRICS], step=batch_idx)

    def train_epoch_end(self, output: EpochOutput, metrics: MetricCollection) -> None:
        """Aggregate interesting outputs across all batches."""


class EstimatorForSequenceClassification(SequenceClassificationMixin, Estimator):
    pass


class UncertaintyBasedStrategyForSequenceClassification(SequenceClassificationMixin, UncertaintyBasedStrategy):
    def pool_step(
        self, model: torch.nn.Module, batch: Dict, batch_idx: int, metrics: MetricCollection
    ) -> POOL_BATCH_OUTPUT:
        logits = model(**batch).logits
        return self.score_fn(logits)
