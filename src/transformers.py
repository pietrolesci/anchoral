from typing import Dict, Optional, Union

import torch
from torchmetrics import Metric, MetricCollection
from torchmetrics.classification import Accuracy, F1Score

from src.enums import InputColumns, RunningStage
from src.estimator import Estimator
from src.query_strategies.base import ActiveEstimator
from src.types import BATCH_OUTPUT, EVAL_BATCH_OUTPUT, POOL_BATCH_OUTPUT


class SequenceClassificationMixin:
    def step(self, model: torch.nn.Module, batch: Dict, batch_idx: int, metrics: MetricCollection) -> BATCH_OUTPUT:
        out = model(**batch)
        preds = out.logits.argmax(-1)
        out_metrics = metrics(preds, batch[InputColumns.TARGET])
        return {"loss": out.loss, "logits": out.logits, "metrics": out_metrics}

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

    def configure_metrics(self, stage: Optional[RunningStage] = None) -> Union[MetricCollection, Metric, None]:
        return MetricCollection(
            {
                "accuracy": Accuracy("multiclass", num_classes=self.model.num_labels),
                "f1_macro": F1Score("multiclass", num_classes=self.model.num_labels, average="macro"),
            }
        )

    def log(self, output: BATCH_OUTPUT, batch_idx: int, stage: RunningStage) -> None:
        self.fabric.log("loss", output["loss"], step=batch_idx)
        self.fabric.log_dict(output["metrics"], step=batch_idx)


class EstimatorForSequenceClassification(SequenceClassificationMixin, Estimator):
    pass


class ActiveEstimatorForSequenceClassification(SequenceClassificationMixin, ActiveEstimator):
    def pool_step(
        self, model: torch.nn.Module, batch: Dict, batch_idx: int, metrics: MetricCollection
    ) -> POOL_BATCH_OUTPUT:
        return self.step(model, batch, batch_idx, metrics)
