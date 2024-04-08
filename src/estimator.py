import time
from typing import Optional, Union

import numpy as np
import pandas as pd
from lightning.fabric.wrappers import _FabricDataLoader, _FabricModule
from torchmetrics import MeanMetric, Metric, MetricCollection
from torchmetrics.classification import Accuracy, CalibrationError, F1Score, Precision, Recall, Specificity

from energizer.active_learning.datastores.classification import ActivePandasDataStoreForSequenceClassification
from energizer.enums import InputKeys, OutputKeys, RunningStage, SpecialKeys
from energizer.utilities import ld_to_dl, move_to_cpu

# pyright: reportGeneralTypeIssues=false


def get_metrics(task: str, num_classes: int, average: str, validate_args: bool = True) -> dict[str, Metric]:
    kwargs = {"task": task, "num_classes": num_classes, "average": average, "validate_args": validate_args}
    suffix = "perclass" if average == "none" else average
    return {
        f"accuracy_{suffix}": Accuracy(**kwargs),
        f"f1_{suffix}": F1Score(**kwargs),
        f"precision_{suffix}": Precision(**kwargs),
        f"recall_{suffix}": Recall(**kwargs),
        f"specificity_{suffix}": Specificity(**kwargs),
    }


class SequenceClassificationMixin:
    minority_class_ids: list[int] = None

    def configure_metrics(self, stage: Optional[Union[str, RunningStage]] = None) -> dict[str, MetricCollection]:
        num_classes = self.model.num_labels
        task = "multiclass"
        metrics_dict = {"loss": MetricCollection({"loss": MeanMetric()}, prefix=f"{stage}/")}

        if stage == RunningStage.TEST:
            metrics = MetricCollection(
                {
                    # NOTE: don't need macro as it is a simple average of the per-class scores
                    # NOTE: using validate_args=False as I have already checked correctness and works faster
                    **get_metrics(task, num_classes, "none", validate_args=False),
                    **get_metrics(task, num_classes, "micro", validate_args=False),
                    "ece": CalibrationError(task, num_classes=num_classes, n_bins=10, norm="l1", validate_args=False),
                    "mce": CalibrationError(task, num_classes=num_classes, n_bins=10, norm="max", validate_args=False),
                    "rmsce": CalibrationError(task, num_classes=num_classes, n_bins=10, norm="l2", validate_args=False),
                    # "auroc_perclass": AUROC(task, num_classes=num_classes, average="none", thresholds=None),
                    # "auprc_perclass": AveragePrecision(
                    #     task, num_classes=num_classes, average="none", thresholds=None
                    # ),
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
                {
                    "f1_perclass": F1Score(task, num_classes=num_classes, average="none", validate_args=False),
                    "f1_macro": F1Score(task, num_classes=num_classes, average="macro", validate_args=False),
                },
                prefix=f"{stage}/",
            )

        metrics_dict["metrics"] = metrics

        return {k: v.to(self.device) for k, v in metrics_dict.items()}

    def step(
        self,
        stage: Union[str, RunningStage],
        model: _FabricModule,
        batch: dict,
        batch_idx: int,
        loss_fn,
        metrics: dict[str, MetricCollection],
    ) -> dict:
        uid = batch.pop(InputKeys.ON_CPU)[SpecialKeys.ID]
        out = model(**batch)

        # update metrics
        metrics["metrics"](out.logits.detach(), batch[InputKeys.LABELS])
        metrics["loss"](out.loss.detach())

        return {OutputKeys.LOSS: out.loss, OutputKeys.LOGITS: out.logits, SpecialKeys.ID: uid}

    def epoch_end(self, stage: Union[str, RunningStage], output: list[dict], metrics: dict) -> dict:
        """Aggregate and log metrics after each train/validation/test/pool epoch."""
        # hack
        assert self.minority_class_ids is not None, "You need to set `minority_class_ids` attribute."

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
        minority_keys = [k for k in per_class for i in self.minority_class_ids if k.endswith(f"f1_class{i}")]
        per_class[f"{stage}/avg_f1_minclass"] = np.array([per_class[k] for k in minority_keys]).mean()

        others = {k: v.item() for k, v in metrics_on_cpu["metrics"].items() if "_perclass" not in k}
        loss = {k: v.item() for k, v in metrics_on_cpu["loss"].items()}
        logs = {**per_class, **others, **loss, "summary/budget": self.tracker.global_budget}

        step = self.tracker.global_epoch if stage == RunningStage.TRAIN else self.tracker.global_round
        self.log_dict(logs, step=step)

        return {**out, **logs}

    def on_fit_end(self, *args, **kwargs) -> None:
        logs = {
            "summary/num_optimization_steps": self.tracker.global_step,
            "summary/num_epochs": self.tracker.global_epoch,
        }
        self.log_dict(logs, step=self.tracker.global_round)

    def on_round_start(self, datastore: ActivePandasDataStoreForSequenceClassification) -> None:
        # compute label distribution
        counts: dict[int, int] = dict(datastore.get_by_ids(datastore.get_train_ids())[InputKeys.LABELS].value_counts())
        for i in range(self.model.num_labels):
            if i not in counts:
                counts[i] = 0
        logs = {
            **{f"summary/cumulative_count_class_{k}": v for k, v in counts.items()},
            "summary/labelled_size": datastore.labelled_size(),
            "summary/pool_size": datastore.pool_size(),
        }
        self.log_dict(logs, step=self.tracker.global_round)

    def select_pool_subset(
        self,
        model: _FabricModule,
        loader: _FabricDataLoader,
        datastore: ActivePandasDataStoreForSequenceClassification,
        **kwargs,
    ) -> list[int]:
        subpool_ids = super().select_pool_subset(model, loader, datastore, **kwargs)  # type: ignore
        self.log("summary/subpool_size", len(subpool_ids), step=self.tracker.global_round)  # type: ignore

        # creating this hook
        self.callback("on_select_pool_subset_end", subpool_ids=subpool_ids)

        return subpool_ids

    def search_pool(
        self,
        datastore: ActivePandasDataStoreForSequenceClassification,
        search_query_embeddings: dict[str, np.ndarray],
        search_query_ids: dict[str, list[int]],
    ) -> pd.DataFrame:
        start_time = time.perf_counter()

        search_results = super().search_pool(datastore, search_query_embeddings, search_query_ids)  # type: ignore

        ids_retrieved = search_results[SpecialKeys.ID].tolist()
        logs = {
            "timer/search": time.perf_counter() - start_time,
            "search/ids_retrieved": len(ids_retrieved),
            "search/unique_ids_retrieved": len(set(ids_retrieved)),
            "search/num_search_queries": len(set(search_query_ids)),
        }
        self.log_dict(logs, step=self.tracker.global_round)  # type: ignore

        return search_results

    def select_search_query(
        self, model: _FabricModule, datastore: ActivePandasDataStoreForSequenceClassification, query_size: int, **kwargs
    ) -> list[int]:
        search_query_ids = super().select_search_query(model, datastore, query_size, **kwargs)

        # creating this hook
        self.callback("on_select_search_query_end", search_query_ids=search_query_ids)

        return search_query_ids
