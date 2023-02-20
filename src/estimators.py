from typing import Callable, Dict, MutableMapping, Optional, Union

import numpy as np
import torch
from lightning.fabric.wrappers import _FabricModule
from lightning.pytorch.utilities.parsing import AttributeDict
from torchmetrics import MetricCollection
from torchmetrics.classification import Accuracy, F1Score
from transformers import AutoModelForSequenceClassification

from src.energizer.active_learning.strategies import RandomStrategy, UncertaintyBasedStrategy
from src.energizer.containers import RoundOutput
from src.energizer.enums import InputKeys, OutputKeys, RunningStage, SpecialKeys
from src.energizer.estimator import Estimator
from src.energizer.types import BATCH_OUTPUT, EPOCH_OUTPUT, EVAL_BATCH_OUTPUT, METRIC, POOL_BATCH_OUTPUT, ROUND_OUTPUT
from src.energizer.utilities import ld_to_dl, move_to_cpu


class SequenceClassificationMixin:

    """
    Method forwarding
    """

    def step(
        self,
        loss_fn: Optional[Union[torch.nn.Module, Callable]],
        model: AutoModelForSequenceClassification,
        batch: Dict,
        metrics: MetricCollection,
    ) -> BATCH_OUTPUT:
        unique_ids = batch.pop(InputKeys.ON_CPU)[SpecialKeys.ID]

        # forward pass
        out = model(**batch)

        # compute weighted loss if provided otherwise
        # used the unweighted loss automaticaly computed by transformers
        loss = loss_fn(out.logits, batch[InputKeys.TARGET]) if loss_fn is not None else out.loss

        # compute metrics
        out_metrics = metrics(out.logits, batch[InputKeys.TARGET])

        return {
            OutputKeys.LOSS: loss,
            OutputKeys.LOGITS: out.logits,
            OutputKeys.METRICS: out_metrics,
            SpecialKeys.ID: unique_ids,
        }

    def train_step(
        self,
        loss_fn: Optional[Union[torch.nn.Module, Callable]],
        model: _FabricModule,
        batch: Dict,
        batch_idx: int,
        metrics: MetricCollection,
    ) -> BATCH_OUTPUT:
        return self.step(loss_fn, model, batch, metrics)

    def validation_step(
        self,
        loss_fn: Optional[Union[torch.nn.Module, Callable]],
        model: _FabricModule,
        batch: Dict,
        batch_idx: int,
        metrics: MetricCollection,
    ) -> EVAL_BATCH_OUTPUT:
        return self.step(loss_fn, model, batch, metrics)

    def test_step(
        self,
        loss_fn: Optional[Union[torch.nn.Module, Callable]],
        model: _FabricModule,
        batch: Dict,
        batch_idx: int,
        metrics: MetricCollection,
    ) -> EVAL_BATCH_OUTPUT:
        return self.step(loss_fn, model, batch, metrics)

    def epoch_end(self, output: EPOCH_OUTPUT, metrics: Optional[METRIC], stage: RunningStage) -> Dict:
        """Aggregate."""
        data = ld_to_dl(output)
        return {
            **move_to_cpu(metrics.compute()),
            f"avg_{OutputKeys.LOSS}": round(np.mean(data[OutputKeys.LOSS]), 6),
        }

    def train_epoch_end(self, output: EPOCH_OUTPUT, metrics: Optional[METRIC]) -> EPOCH_OUTPUT:
        return self.epoch_end(output, metrics, RunningStage.TRAIN)

    def validation_epoch_end(self, output: EPOCH_OUTPUT, metrics: Optional[METRIC]) -> EPOCH_OUTPUT:
        return self.epoch_end(output, metrics, RunningStage.VALIDATION)

    def test_epoch_end(self, output: EPOCH_OUTPUT, metrics: Optional[METRIC]) -> EPOCH_OUTPUT:
        return self.epoch_end(output, metrics, RunningStage.TEST)

    def round_end(self, output: RoundOutput) -> ROUND_OUTPUT:
        """Only keep test outputs."""
        return output.test

    """
    Changes
    """

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


class EstimatorForSequenceClassification(SequenceClassificationMixin, Estimator):
    ...


class UncertaintyBasedStrategyForSequenceClassification(SequenceClassificationMixin, UncertaintyBasedStrategy):
    def pool_step(
        self,
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
