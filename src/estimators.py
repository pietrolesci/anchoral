# NOTE: type annotation is in line with what it is actually returned by
# the overridden methods
from typing import Callable, Dict, List, MutableMapping, Optional, Union

import numpy as np
import torch
from lightning.fabric.wrappers import _FabricModule
from lightning.pytorch.utilities.parsing import AttributeDict
from torchmetrics import MetricCollection
from torchmetrics.classification import Accuracy, F1Score, Precision, Recall
from transformers import AutoModelForSequenceClassification

from src.energizer.active_learning.active_estimator import RoundOutput
from src.energizer.active_learning.data import ActiveDataModule
from src.energizer.active_learning.strategies import RandomStrategy, UncertaintyBasedStrategy
from src.energizer.enums import InputKeys, OutputKeys, RunningStage, SpecialKeys
from src.energizer.estimator import Estimator
from src.energizer.types import ROUND_OUTPUT, Dict
from src.energizer.utilities import ld_to_dl, move_to_cpu


def resolve_strategy_name(s: str) -> str:
    if "UncertaintyBasedStrategyForSequenceClassification" in s["_target_"]:
        return s["score_fn"]
    return "random"


class SequenceClassificationMixin:

    """
    Method forwarding
    """

    def train_step(
        self,
        model: _FabricModule,
        batch: Dict,
        batch_idx: int,
        loss_fn: Optional[Union[torch.nn.Module, Callable]],
        metrics: MetricCollection,
    ) -> Dict:
        return self.step(model, batch, batch_idx, loss_fn, metrics, RunningStage.TRAIN)

    def validation_step(
        self,
        model: _FabricModule,
        batch: Dict,
        batch_idx: int,
        loss_fn: Optional[Union[torch.nn.Module, Callable]],
        metrics: MetricCollection,
    ) -> Dict:
        return self.step(model, batch, batch_idx, loss_fn, metrics, RunningStage.VALIDATION)

    def test_step(
        self,
        model: _FabricModule,
        batch: Dict,
        batch_idx: int,
        loss_fn: Optional[Union[torch.nn.Module, Callable]],
        metrics: MetricCollection,
    ) -> Dict:
        return self.step(model, batch, batch_idx, loss_fn, metrics, RunningStage.TEST)

    def train_epoch_end(self, output: List[Dict], metrics: MetricCollection) -> Dict:
        return self.epoch_end(output, metrics, RunningStage.TRAIN)

    def validation_epoch_end(self, output: List[Dict], metrics: MetricCollection) -> Dict:
        return self.epoch_end(output, metrics, RunningStage.VALIDATION)

    def test_epoch_end(self, output: List[Dict], metrics: MetricCollection) -> Dict:
        return self.epoch_end(output, metrics, RunningStage.TEST)

    def pool_epoch_end(self, output: List[Dict], metrics: MetricCollection) -> Dict:
        return self.epoch_end(output, metrics, RunningStage.POOL)

    """
    Actual methods
    """

    def step(
        self,
        model: AutoModelForSequenceClassification,
        batch: Dict,
        batch_idx: int,
        loss_fn: Optional[Union[torch.nn.Module, Callable]],
        metrics: MetricCollection,
        stage: RunningStage,
    ) -> Dict:
        """This serves as train/validation/test step for transformers classifiers."""
        unique_ids = batch.pop(InputKeys.ON_CPU)[SpecialKeys.ID]

        # forward pass
        out = model(**batch)

        # compute weighted loss if provided otherwise use the
        # unweighted loss automaticaly computed by transformers
        loss = loss_fn(out.logits, batch[InputKeys.TARGET]) if loss_fn is not None else out.loss

        # compute metrics
        # NOTE: we do not return metrics since they can be aggregated using `metrics.compute()` later
        out_metrics = metrics(out.logits, batch[InputKeys.TARGET])

        # log batch-level metrics
        if stage == RunningStage.TRAIN:
            # NOTE: only log at the batch level training
            logs = {OutputKeys.LOSS: loss, **out_metrics}
            self.log_dict({f"{stage}/{k}": v for k, v in logs.items()}, step=self.progress_tracker.global_batch)

        return {
            OutputKeys.LOSS: loss,
            OutputKeys.LOGITS: out.logits,
            SpecialKeys.ID: unique_ids,
            OutputKeys.METRICS: out_metrics,
        }

    def epoch_end(self, output: List[Dict], metrics: MetricCollection, stage: RunningStage) -> Dict:
        """Aggregate and log metrics after each train/validation/test/pool epoch."""
        # print(self.progress_tracker.current_stage, self.progress_tracker.round_tracker.current)

        data = ld_to_dl(output)

        # aggregate instance-level metrics
        logits = np.concatenate(data.pop(OutputKeys.LOGITS))
        unique_ids = np.concatenate(data.pop(SpecialKeys.ID))
        out = {OutputKeys.LOGITS: logits, SpecialKeys.ID: unique_ids}
        if stage == RunningStage.POOL:
            out[OutputKeys.SCORES] = np.concatenate(data.pop(OutputKeys.SCORES))
            return out

        # aggregate and log epoch-level metrics
        aggregated_metrics = move_to_cpu(metrics.compute())  # NOTE: metrics are still on device
        aggregated_loss = round(np.mean(data[OutputKeys.LOSS]), 6)
        logs = {OutputKeys.LOSS: aggregated_loss, **aggregated_metrics}
        logs = {f"{stage}_end/{k}": v for k, v in logs.items()}
        self.log_dict(logs, step=self.progress_tracker.safe_global_epoch)

        # if active_fit log with budget on the x-axis
        if stage == RunningStage.TEST and hasattr(self.progress_tracker, "global_budget"):
            logs = {f"{k}_vs_budget": v for k, v in logs.items()}
            self.log_dict(logs, step=self.progress_tracker.global_budget)

        return {
            OutputKeys.LOSS: aggregated_loss,
            OutputKeys.METRICS: aggregated_metrics,
            **out,
        }

    def round_epoch_end(self, output: RoundOutput, datamodule: ActiveDataModule) -> ROUND_OUTPUT:
        """Log round-level statistics."""
        # print(
        #     "END",
        #     self.progress_tracker.current_stage,
        #     self.progress_tracker.round_tracker.current,
        #     datamodule.data_statistics(self.progress_tracker.global_round),
        # )
        logs = {
            "max_epochs": self.progress_tracker.epoch_tracker.max,
            "num_train_batches": self.progress_tracker.train_tracker.max,
            "num_validation_batches": self.progress_tracker.validation_tracker.max,
            "global_train_steps": self.progress_tracker.step_tracker.total,
            **datamodule.data_statistics(self.progress_tracker.global_round),
        }
        logs = {f"round_stats/{k}": v for k, v in logs.items()}
        self.log_dict(logs, step=self.progress_tracker.global_round)

        return output

    def active_fit_end(self, output: List[ROUND_OUTPUT]) -> Dict:
        """Log metrics at the end of training."""
        logs = ld_to_dl([out.test[OutputKeys.METRICS] for out in output])
        return {
            **{f"hparams/test_{k}": v[-1].item() for k, v in logs.items()},
            **{f"hparams/test_{k}_auc": np.trapz(v) for k, v in logs.items()},
        }

    """
    Changes
    """

    @property
    def hparams(self) -> Union[AttributeDict, MutableMapping]:
        hparams = super().hparams
        hparams["name_or_path"] = self.model.name_or_path
        return hparams

    def configure_metrics(self, stage: Optional[RunningStage] = None) -> Optional[MetricCollection]:
        if stage == RunningStage.POOL:
            return
        num_classes = self.model.num_labels
        task = "multiclass"

        # NOTE: you are in charge of moving it to the correct device
        return MetricCollection(
            {
                "accuracy": Accuracy(task, num_classes=num_classes),
                "f1_macro": F1Score(task, num_classes=num_classes, average="macro"),
                "precision_macro": Precision(task, num_classes=num_classes, average="macro"),
                "recall_macro": Recall(task, num_classes=num_classes, average="macro"),
                "f1_micro": F1Score(task, num_classes=num_classes, average="micro"),
                "precision_micro": Precision(task, num_classes=num_classes, average="micro"),
                "recall_micro": Recall(task, num_classes=num_classes, average="micro"),
            }
        ).to(self.device)


# normal training
class EstimatorForSequenceClassification(SequenceClassificationMixin, Estimator):
    ...


# random strategy
class RandomStrategyForSequenceClassification(SequenceClassificationMixin, RandomStrategy):
    ...


# uncertainty strategy
class UncertaintyBasedStrategyForSequenceClassification(SequenceClassificationMixin, UncertaintyBasedStrategy):
    def pool_step(
        self,
        model: _FabricModule,
        batch: Dict,
        batch_idx: int,
        metrics: Optional[MetricCollection] = None,
    ) -> Dict:
        _ = batch.pop(InputKeys.ON_CPU)  # this is already handled in the `evaluation_step`

        logits = model(**batch).logits
        scores = self.score_fn(logits)

        return {OutputKeys.SCORES: scores, OutputKeys.LOGITS: logits}
