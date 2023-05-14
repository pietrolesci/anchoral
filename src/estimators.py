from typing import Dict, List, Optional, Union

import numpy as np
from lightning.fabric.wrappers import _FabricModule
from torchmetrics import MetricCollection
from torchmetrics.classification import Accuracy, F1Score, Precision, Recall

from energizer.enums import InputKeys, OutputKeys, RunningStage, SpecialKeys
from energizer.types import ROUND_OUTPUT
from energizer.utilities import ld_to_dl, move_to_cpu


class SequenceClassificationMixin:
    def configure_metrics(self, *_) -> MetricCollection:
        num_classes = self.model.num_labels  # type: ignore
        task = "multiclass"
        metrics = MetricCollection(
            {
                "accuracy_macro": Accuracy(task, num_classes=num_classes, average="macro"),
                "f1_macro": F1Score(task, num_classes=num_classes, average="macro"),
                "precision_macro": Precision(task, num_classes=num_classes, average="macro"),
                "recall_macro": Recall(task, num_classes=num_classes, average="macro"),
                # "average_precision_macro": AveragePrecision(task, num_classes=num_classes, average="macro", warn_only=True),
                # "auroc": AUROC(task, num_classes=num_classes, average="macro", warn_only=True),
                "accuracy_micro": Accuracy(task, num_classes=num_classes, average="micro"),
                "f1_micro": F1Score(task, num_classes=num_classes, average="micro"),
                "precision_micro": Precision(task, num_classes=num_classes, average="micro"),
                "recall_micro": Recall(task, num_classes=num_classes, average="micro"),
            }
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

        on_cpu = batch.pop(InputKeys.ON_CPU, None)
        out = model(**batch)
        out_metrics = metrics(out.logits, batch[InputKeys.TARGET])

        if stage == RunningStage.TRAIN:
            logs = {OutputKeys.LOSS: out.loss, **out_metrics}
            self.log_dict({f"{stage}/{k}": v for k, v in logs.items()}, step=self.progress_tracker.global_batch)  # type: ignore

        output = {
            OutputKeys.LOSS: out.loss,
            OutputKeys.LOGITS: out.logits,
        }
        if on_cpu is not None and SpecialKeys.ID in on_cpu:
            output[SpecialKeys.ID] = on_cpu[SpecialKeys.ID]  # type: ignore
        return output

    def epoch_end(self, stage: Union[str, RunningStage], output: List[Dict], metrics: MetricCollection) -> Dict:
        """Aggregate and log metrics after each train/validation/test/pool epoch."""

        data = ld_to_dl(output)

        # aggregate instance-level metrics
        out = {OutputKeys.LOGITS: np.concatenate(data.pop(OutputKeys.LOGITS))}

        if SpecialKeys.ID in data:
            out[SpecialKeys.ID] = np.concatenate(data.pop(SpecialKeys.ID))  # type: ignore

        if stage == RunningStage.POOL:
            out[OutputKeys.SCORES] = np.concatenate(data.pop(OutputKeys.SCORES))
            return out

        # aggregate and log epoch-level metrics
        aggregated_metrics = move_to_cpu(metrics.compute())  # NOTE: metrics are still on device
        aggregated_loss = round(np.mean(data[OutputKeys.LOSS]), 6)
        logs = {OutputKeys.LOSS: aggregated_loss, **aggregated_metrics}
        logs = {f"{stage}_end/{k}": v for k, v in logs.items()}
        self.log_dict(logs, step=self.progress_tracker.safe_global_epoch)  # type: ignore

        # if active_fit log with budget on the x-axis
        if stage == RunningStage.TEST and hasattr(self.progress_tracker, "global_budget"):  # type: ignore
            logs = {f"{k}_vs_budget": v for k, v in logs.items()}
            self.log_dict(logs, step=self.progress_tracker.global_budget)  # type: ignore

        return {OutputKeys.LOSS: aggregated_loss, **out, **aggregated_metrics}

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
