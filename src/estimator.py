from typing import Dict, List, Optional, Union

import numpy as np
from lightning.fabric.wrappers import _FabricModule
from torchmetrics import MeanMetric, MetricCollection
from torchmetrics.classification import AUROC, Accuracy, AveragePrecision, F1Score, Precision, Recall

from energizer.enums import InputKeys, OutputKeys, RunningStage, SpecialKeys
from energizer.utilities import ld_to_dl, move_to_cpu


class SequenceClassificationMixin:
    def configure_metrics(self, stage: Optional[Union[str, RunningStage]] = None) -> MetricCollection:
        num_classes = self.model.num_labels  # type: ignore
        task = "multiclass"

        metrics = MetricCollection(
            [
                MetricCollection(
                    {
                        "accuracy": Accuracy(task, num_classes=num_classes, average="none"),
                        "f1": F1Score(task, num_classes=num_classes, average="none"),
                        "precision": Precision(task, num_classes=num_classes, average="none"),
                        "recall": Recall(task, num_classes=num_classes, average="none"),
                    },
                    postfix="_class",
                ),
                MetricCollection(
                    {
                        "accuracy": Accuracy(task, num_classes=num_classes, average="micro"),
                        "f1": F1Score(task, num_classes=num_classes, average="micro"),
                        "precision": Precision(task, num_classes=num_classes, average="micro"),
                        "recall": Recall(task, num_classes=num_classes, average="micro"),
                    },
                    postfix="_micro",
                ),
                MetricCollection(
                    {
                        "accuracy": Accuracy(task, num_classes=num_classes, average="macro"),
                        "f1": F1Score(task, num_classes=num_classes, average="macro"),
                        "precision": Precision(task, num_classes=num_classes, average="macro"),
                        "recall": Recall(task, num_classes=num_classes, average="macro"),
                    },
                    postfix="_macro",
                ),
                MetricCollection(
                    {
                        "loss": MeanMetric(),
                        "auroc": AUROC("multiclass", thresholds=30, num_classes=num_classes, average="macro"),
                        "ap": AveragePrecision("multiclass", thresholds=30, num_classes=num_classes, average="macro"),
                    }
                ),
            ],  # type: ignore
            prefix=f"{stage}/",
        )
        return metrics.to(self.device)  # type: ignore

    def step(
        self,
        stage: Union[str, RunningStage],
        model: _FabricModule,
        batch: Dict,
        batch_idx: int,
        loss_fn,
        metrics: MetricCollection,
    ) -> Dict:

        uid = batch.pop(InputKeys.ON_CPU)[SpecialKeys.ID]
        out = model(**batch)

        # update metrics
        metrics(preds=out.logits, target=batch[InputKeys.TARGET], value=out.loss.detach())

        return {
            OutputKeys.LOSS: out.loss,
            OutputKeys.LOGITS: out.logits,
            SpecialKeys.ID: uid,
        }

    def epoch_end(self, stage: Union[str, RunningStage], output: List[Dict], metrics: MetricCollection) -> Dict:
        """Aggregate and log metrics after each train/validation/test/pool epoch."""

        data = ld_to_dl(output)
        data.pop(OutputKeys.LOSS, None)  # this is already part of the metrics

        out = {k: np.concatenate(v) for k, v in data.items()}
        _metrics = move_to_cpu(metrics.compute())

        # flatten per-class metric
        per_class = {f"{k}{idx}": v[idx] for k, v in _metrics.items() for idx in range(v.size) if k.endswith("_class")}
        micro = {k: v for k, v in _metrics.items() if not k.endswith("_class")}
        _metrics = {**per_class, **micro}

        # log
        self.log_dict(_metrics, step=self.tracker.safe_global_epoch)  # type: ignore
        if stage == RunningStage.TEST and hasattr(self.tracker, "global_budget"):  # type: ignore
            self.log_dict({f"{k}_vs_budget": v for k, v in _metrics.items()}, step=self.tracker.global_budget)  # type: ignore

        return {**out, **_metrics}
