from typing import Callable, Dict, MutableMapping, Optional, Union

import numpy as np
import torch
from lightning.fabric.wrappers import _FabricModule
from lightning.pytorch.utilities.parsing import AttributeDict
from torchmetrics import MetricCollection
from torchmetrics.classification import Accuracy, F1Score
from transformers import AutoModelForSequenceClassification

from src.active_learning.strategies import RandomStrategy, UncertaintyBasedStrategy
from src.enums import InputKeys, OutputKeys, RunningStage, SpecialKeys
from src.estimator import Estimator
from src.types import BATCH_OUTPUT, EPOCH_OUTPUT, EVAL_BATCH_OUTPUT, POOL_BATCH_OUTPUT
from src.utilities import move_to_cpu


class SequenceClassificationMixin:
    @property
    def hparams(self) -> Union[AttributeDict, MutableMapping]:
        hparams = super().hparams
        hparams["name_or_path"] = self.model.name_or_path
        return hparams

    def configure_loss_fn(
        self,
        loss_fn: Optional[Union[str, torch.nn.Module, Callable]],
        loss_fn_kwargs: Optional[Dict],
        stage: RunningStage,
    ) -> Optional[Union[torch.nn.Module, Callable]]:
        if loss_fn_kwargs is not None and "weight" in loss_fn_kwargs:
            loss_fn_kwargs["weight"] = torch.tensor(loss_fn_kwargs["weight"], dtype=torch.float32, device=self.device)

        return super().configure_loss_fn(loss_fn, loss_fn_kwargs, stage)

    def configure_metrics(self, stage: Optional[RunningStage] = None) -> Optional[MetricCollection]:
        if stage == RunningStage.POOL:
            return
        # you are in charge of moving it to the correct device
        return MetricCollection(
            {
                "accuracy": Accuracy("multiclass", num_classes=self.model.num_labels),
                "f1_macro": F1Score("multiclass", num_classes=self.model.num_labels, average="macro"),
            }
        ).to(self.device)

    def train_step(
        self,
        loss_fn: Optional[Union[torch.nn.Module, Callable]],
        model: _FabricModule,
        batch: Dict,
        batch_idx: int,
        metrics: MetricCollection,
    ) -> BATCH_OUTPUT:
        return self.step(loss_fn, model, batch, batch_idx, metrics)

    def validation_step(
        self,
        loss_fn: Optional[Union[torch.nn.Module, Callable]],
        model: _FabricModule,
        batch: Dict,
        batch_idx: int,
        metrics: MetricCollection,
    ) -> EVAL_BATCH_OUTPUT:
        return self.step(loss_fn, model, batch, batch_idx, metrics)

    def test_step(
        self,
        loss_fn: Optional[Union[torch.nn.Module, Callable]],
        model: _FabricModule,
        batch: Dict,
        batch_idx: int,
        metrics: MetricCollection,
    ) -> EVAL_BATCH_OUTPUT:
        return self.step(loss_fn, model, batch, batch_idx, metrics)

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

    def train_epoch_end(self, output: EPOCH_OUTPUT, metrics: MetricCollection) -> EPOCH_OUTPUT:
        return self.epoch_end(output, metrics, RunningStage.TRAIN)

    def validation_epoch_end(self, output: EPOCH_OUTPUT, metrics: MetricCollection) -> EPOCH_OUTPUT:
        return self.epoch_end(output, metrics, RunningStage.VALIDATION)

    def test_epoch_end(self, output: EPOCH_OUTPUT, metrics: MetricCollection) -> EPOCH_OUTPUT:
        return self.epoch_end(output, metrics, RunningStage.TEST)

    """
    Aggregation and logging
    """

    def epoch_end(self, output: EPOCH_OUTPUT, metrics: MetricCollection, stage: RunningStage) -> EPOCH_OUTPUT:
        if stage != RunningStage.TRAIN:
            logs = {
                **move_to_cpu(metrics.compute()),
                "avg_loss": round(np.mean([out[OutputKeys.LOSS] for out in output]), 6),
            }

            logs = {f"{stage}_end/{k}": v for k, v in logs.items()}
            step = self.epoch_level_logging_step(stage)
            self.fabric.log_dict(logs, step=step)
            # print("EPOCH", stage, step, flush=True)

            return logs

    def log(self, output: BATCH_OUTPUT, batch_idx: int, stage: RunningStage) -> None:
        logs = {OutputKeys.LOSS: output[OutputKeys.LOSS], **output[OutputKeys.METRICS]}
        logs = {f"{stage}/{k}": v for k, v in logs.items()}
        step = self.batch_level_logging_step(stage, batch_idx)
        # print("BATCH", stage, step, flush=True)
        self.fabric.log_dict(logs, step=step)

    def batch_level_logging_step(self, stage: RunningStage, batch_idx: int) -> int:
        return getattr(self.counter, f"num_{stage}_batches")
        # if stage == RunningStage.TRAIN:
        #     return self.counter.num_steps
        # elif stage == RunningStage.VALIDATION:
        #     if self.counter.num_epochs > 0 and batch_idx == 0:
        #         return self.counter.num_epochs + 1
        #     return batch_idx * (self.counter.num_epochs + 1)
        # elif stage == RunningStage.TEST:
        #     return batch_idx 
     
    def epoch_level_logging_step(self, stage: RunningStage) -> int:
        return self.counter.num_epochs


class EstimatorForSequenceClassification(SequenceClassificationMixin, Estimator):
    ...


class ActiveLearningLoggingMixin:
    
    def batch_level_logging_step(self, stage: RunningStage, batch_idx: int) -> int:
        return getattr(self.counter, f"total_{stage}_batches") + getattr(self.counter, f"num_{stage}_batches")
        # if stage == RunningStage.TRAIN:
        #     step = self.counter.num_steps
        # elif stage == RunningStage.VALIDATION:
        #     step = batch_idx * (self.counter.num_epochs + 1)
        # elif stage == RunningStage.TEST:
        #     step = batch_idx 
        # return (self.counter.num_rounds + 1) * step

    def epoch_level_logging_step(self, stage: RunningStage) -> int:
        # # step arithmetic
        # if stage == RunningStage.TEST:
        #     step = self.counter.num_rounds
        # elif stage == RunningStage.VALIDATION:
        #     step = (self.counter.num_rounds + 1) * self.num_epochs
        return getattr(self.counter, "total_epochs") + getattr(self.counter, "num_epochs")



class UncertaintyBasedStrategyForSequenceClassification(
    ActiveLearningLoggingMixin, SequenceClassificationMixin, UncertaintyBasedStrategy
):
    def pool_step(
        self,
        loss_fn: Optional[Union[torch.nn.Module, Callable]],
        model: _FabricModule,
        batch: Dict,
        batch_idx: int,
        metrics: Optional[MetricCollection] = None,
    ) -> POOL_BATCH_OUTPUT:
        _ = batch.pop(InputKeys.ON_CPU)  # this is already handled in the `eval_batch_loop`

        logits = model(**batch).logits
        scores = self.score_fn(logits)

        return {OutputKeys.SCORES: scores, OutputKeys.LOGITS: logits}


class RandomStrategyForSequenceClassification(SequenceClassificationMixin, RandomStrategy):
    pass
