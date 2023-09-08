from typing import Dict, List, Optional, Union

import numpy as np
from lightning.fabric.wrappers import _FabricModule
from torchmetrics import BootStrapper, MeanMetric, Metric, MetricCollection
from torchmetrics.classification import AUROC, Accuracy, AveragePrecision, F1Score, Precision, Recall, Specificity

from energizer.enums import InputKeys, OutputKeys, RunningStage, SpecialKeys
from energizer.utilities import ld_to_dl, move_to_cpu

# from torchmetrics.wrappers import BootStrapper


# pyright: reportGeneralTypeIssues=false


def get_metrics(num_classes: int, average: str) -> Dict[str, Metric]:
    task = "multiclass"
    suffix = "perclass" if average == "none" else average
    return {
        f"accuracy_{suffix}": Accuracy(task, num_classes=num_classes, average=average),
        f"f1_{suffix}": F1Score(task, num_classes=num_classes, average=average),
        f"precision_{suffix}": Precision(task, num_classes=num_classes, average=average),
        f"recall_{suffix}": Recall(task, num_classes=num_classes, average=average),
        f"specificity_{suffix}": Specificity(task, num_classes=num_classes, average=average),
    }


class SequenceClassificationMixin:
    def configure_metrics(self, stage: Optional[Union[str, RunningStage]] = None) -> Dict:
        num_classes = self.model.num_labels

        metrics = {
            "metrics": MetricCollection(
                {
                    **get_metrics(num_classes, "none"),
                    **get_metrics(num_classes, "micro"),
                    "auprc_perclass": AveragePrecision(num_classes, average="none", thresholds=None),
                    "auroc_perclass": AUROC(num_classes, average="none", thresholds=None)
                    # don't need macro as it is a simple average of the per-class scores
                    # **get_metrics(num_classes, "macro"),
                },
                prefix=f"{stage}/",
            ),
            "loss": MeanMetric(),
            "bootstrapped_f1": BootStrapper(
                F1Score("multiclass", num_classes=num_classes, average="none"), num_bootstraps=200
            ),
        }

        return {k: v.to(self.device) for k, v in metrics.items()}

    def step(
        self,
        stage: Union[str, RunningStage],
        model: _FabricModule,
        batch: Dict,
        batch_idx: int,
        loss_fn,
        metrics: Dict,
    ) -> Dict:

        uid = batch.pop(InputKeys.ON_CPU)[SpecialKeys.ID]
        out = model(**batch)

        # update metrics
        preds, target, loss = out.logits, batch[InputKeys.TARGET], out.loss.detach()
        metrics["metrics"](preds, target)
        metrics["loss"](loss)
        # metrics["bootstrapped_f1"](preds, target)
        # metrics["bootstrapped_acc"](preds, target)

        return {
            OutputKeys.LOSS: out.loss,
            OutputKeys.LOGITS: out.logits,
            SpecialKeys.ID: uid,
        }

    def epoch_end(self, stage: Union[str, RunningStage], output: List[Dict], metrics: Dict) -> Dict:
        """Aggregate and log metrics after each train/validation/test/pool epoch."""

        data = ld_to_dl(output)
        data.pop(OutputKeys.LOSS, None)  # this is already part of the metrics

        out = {k: np.concatenate(v) for k, v in data.items()}

        # compute metrics
        _metrics = {k: move_to_cpu(v.compute()) for k, v in metrics.items()}

        # log
        per_class = {
            f"{k}{idx}": v[idx] for k, v in _metrics["metrics"].items() for idx in range(v.size) if k.endswith("_class")
        }
        others = {k: v for k, v in _metrics["metrics"].items() if not k.endswith("_class")}
        loss = {"loss": _metrics["loss"]}
        # bootstrapped_f1 = {
        #     f"bootf1_{k}{idx}": v[idx] for k, v in _metrics["bootstrapped_f1"].items() for idx in range(v.size)
        # }
        # bootstrapped_acc = {
        #     f"bootaccuracy_{k}{idx}": v[idx] for k, v in _metrics["bootstrapped_acc"].items() for idx in range(v.size)
        # }
        _metrics = {**per_class, **others, **loss}

        self.log_dict(_metrics, step=self.tracker.safe_global_epoch)
        if stage == RunningStage.TEST and hasattr(self.tracker, "global_budget"):
            self.log_dict({f"{k}_vs_budget": v for k, v in _metrics.items()}, step=self.tracker.global_budget)

        return {**out, **_metrics}
