from pprint import pprint as print
from typing import Dict, List, Optional, Union

import numpy as np
from lightning.fabric.wrappers import _FabricModule
from torchmetrics import MeanMetric, MetricCollection
from torchmetrics.classification import AUROC, Accuracy, F1Score, Precision, Recall

from energizer.enums import InputKeys, OutputKeys, RunningStage, SpecialKeys
from energizer.types import ROUND_OUTPUT
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
                        "loss": MeanMetric(),
                        "auroc": AUROC("multiclass", thresholds=20, num_classes=num_classes, average="macro"),
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
        self.log_dict(_metrics, step=self.progress_tracker.safe_global_epoch)  # type: ignore
        if stage == RunningStage.TEST and hasattr(self.progress_tracker, "global_budget"):  # type: ignore
            self.log_dict({f"{k}_vs_budget": v for k, v in _metrics.items()}, step=self.progress_tracker.global_budget)  # type: ignore

        return {**out, **_metrics}

    # def round_epoch_end(self, output: Dict, *args, **kwargs) -> ROUND_OUTPUT:
    #     """Log round-level statistics."""
    #     logs = {
    #         "max_epochs": self.progress_tracker.epoch_tracker.max,  # type: ignore
    #         "num_train_batches": self.progress_tracker.train_tracker.max,  # type: ignore
    #         "num_validation_batches": self.progress_tracker.validation_tracker.max,  # type: ignore
    #         "global_train_steps": self.progress_tracker.step_tracker.total,  # type: ignore
    #     }
    #     logs = {f"round_stats/{k}": v for k, v in logs.items()}
    #     self.log_dict(logs, step=self.progress_tracker.global_round)  # type: ignore

    #     return output

    # def active_fit_end(self, output: List[ROUND_OUTPUT]) -> Dict:
    #     """Log metrics at the end of training."""
    #     logs = ld_to_dl([out[RunningStage.TEST][OutputKeys.METRICS] for out in output])
    #     return {
    #         **{f"hparams/test_{k}": v[-1].item() for k, v in logs.items()},
    #         **{f"hparams/test_{k}_auc": np.trapz(v) for k, v in logs.items()},
    #     }

    def pool_step(
        self,
        model: _FabricModule,
        batch: Dict,
        batch_idx: int,
        metrics: Optional[MetricCollection] = None,
    ) -> Dict:
        _ = batch.pop(InputKeys.ON_CPU)  # this is already handled in the `evaluation_step`
        logits = model(**batch).logits
        scores = self.score_fn(logits)  # type: ignore

        return {OutputKeys.SCORES: scores, OutputKeys.LOGITS: logits}
