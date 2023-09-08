from typing import Dict, List, Optional, Union

import numpy as np
from lightning.fabric.wrappers import _FabricModule
from torchmetrics import BootStrapper, MeanMetric, Metric, MetricCollection
from torchmetrics.classification import AUROC, Accuracy, AveragePrecision, F1Score, Precision, Recall, Specificity

from energizer.active_learning.datastores.classification import ActivePandasDataStoreForSequenceClassification
from energizer.enums import InputKeys, OutputKeys, RunningStage, SpecialKeys
from energizer.utilities import ld_to_dl, move_to_cpu

# from torchmetrics.wrappers import BootStrapper


# pyright: reportGeneralTypeIssues=false


def get_metrics(task: str, num_classes: int, average: str) -> Dict[str, Metric]:
    suffix = "perclass" if average == "none" else average
    return {
        f"accuracy_{suffix}": Accuracy(task, num_classes=num_classes, average=average),
        f"f1_{suffix}": F1Score(task, num_classes=num_classes, average=average),
        f"precision_{suffix}": Precision(task, num_classes=num_classes, average=average),
        f"recall_{suffix}": Recall(task, num_classes=num_classes, average=average),
        f"specificity_{suffix}": Specificity(task, num_classes=num_classes, average=average),
    }


class SequenceClassificationMixin:
    def configure_metrics(self, stage: Optional[Union[str, RunningStage]] = None) -> Dict[str, MetricCollection]:
        num_classes = self.model.num_labels
        task = "multiclass"

        metrics_dict = {"loss": MetricCollection({"loss": MeanMetric()}, prefix=f"{stage}/")}

        if stage == RunningStage.TEST:

            metrics = MetricCollection(
                {
                    # NOTE: don't need macro as it is a simple average of the per-class scores
                    **get_metrics(task, num_classes, "none"),
                    **get_metrics(task, num_classes, "micro"),
                    # "auprc_perclass": AveragePrecision(task, num_classes=num_classes, average="none", thresholds=None),
                    # "auroc_perclass": AUROC(task, num_classes=num_classes, average="none", thresholds=None),
                    # "bootf1_perclass": BootStrapper(
                    #     F1Score(task, num_classes=num_classes, average="none"), num_bootstraps=1_000,
                    # ),
                    # "bootf1_micro": BootStrapper(
                    #     F1Score(task, num_classes=num_classes, average="micro"), num_bootstraps=1
                    # ),
                },
                prefix=f"{stage}/",
            )
        else:
            metrics = MetricCollection(
                {"f1_perclass": F1Score(task, num_classes=num_classes, average="none")}, prefix=f"{stage}/"
            )

        metrics_dict["metrics"] = metrics

        return {k: v.to(self.device) for k, v in metrics_dict.items()}

    def step(
        self,
        stage: Union[str, RunningStage],
        model: _FabricModule,
        batch: Dict,
        batch_idx: int,
        loss_fn,
        metrics: Dict[str, MetricCollection],
    ) -> Dict:

        uid = batch.pop(InputKeys.ON_CPU)[SpecialKeys.ID]
        out = model(**batch)

        # update metrics
        metrics["metrics"](out.logits.detach(), batch[InputKeys.TARGET])
        metrics["loss"](out.loss.detach())

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

        # compute metrics and move to cpu
        metrics_on_cpu = {k: move_to_cpu(v.compute()) for k, v in metrics.items()}

        # collect logs and modify the key for each metrics
        per_class = {
            f"{k.replace('_perclass', '')}_class{idx}": v[idx].item()
            for k, v in metrics_on_cpu["metrics"].items()
            for idx in range(v.size)
            if "_perclass" in k
        }
        others = {k: v.item() for k, v in metrics_on_cpu["metrics"].items() if "_perclass" not in k}
        loss = {k: v.item() for k, v in metrics_on_cpu["loss"].items()}
        logs = {**per_class, **others, **loss, "summary/budget": self.tracker.global_budget}

        self.log_dict(logs, step=self.tracker.global_round)

        return {**out, **logs}

    def round_start(self, datastore: ActivePandasDataStoreForSequenceClassification) -> None:
        # log initial statistics
        # if self.tracker.global_round == 0:
        logs = _get_logs(datastore, self.model.num_labels)
        self.log_dict(logs, step=self.tracker.global_round)

    # def round_end(self, datastore: ActivePandasDataStoreForSequenceClassification, output: Dict) -> Dict:
    #     logs = _get_logs(datastore, self.model.num_labels)
    #     self.log_dict(logs, step=self.tracker.global_round)


def _get_logs(datastore: ActivePandasDataStoreForSequenceClassification, num_classes: int) -> Dict:
    # compute label distribution
    counts: Dict[int, int] = dict(datastore.get_by_ids(datastore.get_train_ids())[InputKeys.TARGET].value_counts())
    for i in range(num_classes):
        if i not in counts:
            counts[i] = 0
    return {
        **{f"summary/cumulative_count_class_{k}": v for k, v in counts.items()},
        "summary/labelled_size": datastore.labelled_size(),
        "summary/pool_size": datastore.pool_size(),
    }
