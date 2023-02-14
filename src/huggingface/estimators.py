from typing import Callable, Dict, MutableMapping, Optional, Union

import numpy as np
import torch
from lightning.pytorch.utilities.parsing import AttributeDict
from torchmetrics import MetricCollection
from torchmetrics.classification import Accuracy, F1Score
from transformers import AutoModelForSequenceClassification

from src.enums import InputKeys, OutputKeys, RunningStage, SpecialKeys
from src.estimator import Estimator
from src.query_strategies.base import UncertaintyBasedStrategy
from src.types import BATCH_OUTPUT, EPOCH_OUTPUT, EVAL_BATCH_OUTPUT, POOL_BATCH_OUTPUT
from src.utilities import move_to_cpu


class SequenceClassificationMixin:
    def step(
        self,
        loss_fn: Optional[Union[torch.nn.Module, Callable]],
        model: AutoModelForSequenceClassification,
        batch: Dict,
        batch_idx: int,
        metrics: MetricCollection,
    ) -> BATCH_OUTPUT:
        unique_ids = batch.pop(InputKeys.ON_CPU)[SpecialKeys.ID]

        # forward pass
        out = model(**batch)

        # compute weighted loss if provided otherwise
        # used the unweighted loss automaticaly computed by transformers
        loss = loss_fn(out.logits, batch[InputKeys.TARGET]) if loss_fn is not None else out.loss

        # compute metrics
        preds = out.logits.argmax(-1)
        out_metrics = metrics(preds, batch[InputKeys.TARGET])

        return {
            OutputKeys.LOSS: loss,
            OutputKeys.LOGITS: out.logits,
            OutputKeys.METRICS: out_metrics,
            SpecialKeys.ID: unique_ids,
        }

    @property
    def hparams(self) -> Union[AttributeDict, MutableMapping]:
        hparams = super().hparams
        hparams["name_or_path"] = self.model.name_or_path
        return hparams

    def train_step(
        self,
        loss_fn: Optional[Union[torch.nn.Module, Callable]],
        model: torch.nn.Module,
        batch: Dict,
        batch_idx: int,
        metrics: MetricCollection,
    ) -> BATCH_OUTPUT:
        return self.step(loss_fn, model, batch, batch_idx, metrics)

    def validation_step(
        self,
        loss_fn: Optional[Union[torch.nn.Module, Callable]],
        model: torch.nn.Module,
        batch: Dict,
        batch_idx: int,
        metrics: MetricCollection,
    ) -> EVAL_BATCH_OUTPUT:
        return self.step(loss_fn, model, batch, batch_idx, metrics)

    def test_step(
        self,
        loss_fn: Optional[Union[torch.nn.Module, Callable]],
        model: torch.nn.Module,
        batch: Dict,
        batch_idx: int,
        metrics: MetricCollection,
    ) -> EVAL_BATCH_OUTPUT:
        return self.step(loss_fn, model, batch, batch_idx, metrics)

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
        logs = {OutputKeys.LOSS: output[OutputKeys.LOSS], **output[OutputKeys.METRICS]}
        logs = {f"{stage}/{k}": v for k, v in logs.items()}
        self.fabric.log_dict(logs, step=self.counter.num_steps if stage == RunningStage.TRAIN else batch_idx)

    def epoch_end(self, output: EPOCH_OUTPUT, metrics: MetricCollection, stage: RunningStage) -> EPOCH_OUTPUT:
        if stage == RunningStage.TRAIN:
            return

        logs = {
            **move_to_cpu(metrics.compute()),
            "avg_loss": round(np.mean([out[OutputKeys.LOSS] for out in output]), 6),
        }

        logs = {f"{stage}_end/{k}": v for k, v in logs.items()}
        self.fabric.log_dict(logs, step=self.counter.num_epochs)

        return logs

    def train_epoch_end(self, output: EPOCH_OUTPUT, metrics: MetricCollection) -> EPOCH_OUTPUT:
        return self.epoch_end(output, metrics, RunningStage.TRAIN)

    def validation_epoch_end(self, output: EPOCH_OUTPUT, metrics: MetricCollection) -> EPOCH_OUTPUT:
        return self.epoch_end(output, metrics, RunningStage.VALIDATION)

    def test_epoch_end(self, output: EPOCH_OUTPUT, metrics: MetricCollection) -> EPOCH_OUTPUT:
        return self.epoch_end(output, metrics, RunningStage.TEST)

    def configure_loss_fn(
        self,
        loss_fn: Optional[Union[str, torch.nn.Module, Callable]],
        loss_fn_kwargs: Optional[Dict],
        stage: RunningStage,
    ) -> Optional[Union[torch.nn.Module, Callable]]:
        if loss_fn_kwargs is not None and "weight" in loss_fn_kwargs:
            loss_fn_kwargs["weight"] = torch.tensor(loss_fn_kwargs["weight"], dtype=torch.float32, device=self.device)

        return super().configure_loss_fn(loss_fn, loss_fn_kwargs, stage)


class EstimatorForSequenceClassification(SequenceClassificationMixin, Estimator):
    pass


# class UncertaintyBasedStrategyForSequenceClassification(SequenceClassificationMixin, UncertaintyBasedStrategy):
#     def pool_step(
#         self, model: torch.nn.Module, batch: Dict, batch_idx: int, metrics: MetricCollection
#     ) -> POOL_BATCH_OUTPUT:
#         logits = model(**batch).logits
#         return self.score_fn(logits)
