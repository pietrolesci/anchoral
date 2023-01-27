from typing import Dict, Optional, Union

import torch
from torchmetrics import Metric, MetricCollection
from torchmetrics.classification import Accuracy, F1Score

from src.enums import InputColumns, RunningStage
from src.query_strategies.active_estimator import PoolBasedActiveEstimator
from src.registries import SCORE_FUNCTIONS
from src.types import BATCH_OUTPUT, EVAL_BATCH_OUTPUT, POOL_BATCH_OUTPUT


class ActiveEstimatorForSequenceClassification(PoolBasedActiveEstimator):
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

    def configure_metrics(self, stage: Optional[RunningStage] = None) -> MetricCollection:
        return MetricCollection(
            {
                "accuracy": Accuracy("multiclass", num_classes=self.model.num_labels),
                "f1_macro": F1Score("multiclass", num_classes=self.model.num_labels, average="macro"),
            }
        )

    # def log(self, output: BATCH_OUTPUT, batch_idx: int, stage: RunningStage) -> None:

    #     self.fabric.log("loss", output["loss"], step=batch_idx)
    #     self.fabric.log_dict(output["metrics"], step=batch_idx)


class UncertantyBasedStrategy(ActiveEstimatorForSequenceClassification):
    def __init__(self, score_fn: str, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.score_fn = SCORE_FUNCTIONS.get(score_fn)

    def pool_step(
        self, model: torch.nn.Module, batch: Dict, batch_idx: int, metrics: Optional[MetricCollection] = None
    ) -> POOL_BATCH_OUTPUT:
        out = model(**batch)
        scores = self.score_fn(out.logits)
        return {"scores": scores, "loss": out.loss}
