from typing import Dict, List, Optional

import numpy as np
import torch
from lightning.fabric.wrappers import _FabricModule
from torch.func import functional_call, grad, vmap  # type: ignore
from torchmetrics import MetricCollection
from torchmetrics.classification import Accuracy, F1Score, Precision, Recall
from tqdm.auto import tqdm

from energizer.datastores.base import Datastore
from energizer.enums import InputKeys, OutputKeys, RunningStage, SpecialKeys
from energizer.estimators import Estimator as _Estimator
from energizer.strategies import RandomStrategy as _RandomStrategy
from energizer.strategies import UncertaintyBasedStrategy as _UncertaintyBasedStrategy
from energizer.strategies.uncertainty import UncertaintyBasedStrategySEALS as _UncertaintyBasedStrategySEALS
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
        stage: RunningStage,
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

    def epoch_end(self, stage: RunningStage, output: List[Dict], metrics: MetricCollection) -> Dict:
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

    def active_fit_end(self, output: List[ROUND_OUTPUT]) -> Dict:
        """Log metrics at the end of training."""
        logs = ld_to_dl([out[RunningStage.TEST][OutputKeys.METRICS] for out in output])
        return {
            **{f"hparams/test_{k}": v[-1].item() for k, v in logs.items()},
            **{f"hparams/test_{k}_auc": np.trapz(v) for k, v in logs.items()},
        }


class Estimator(SequenceClassificationMixin, _Estimator):
    ...


class RandomStrategy(SequenceClassificationMixin, _RandomStrategy):
    ...


class UncertaintyBasedStrategy(SequenceClassificationMixin, _UncertaintyBasedStrategy):
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


class UncertaintyBasedStrategySEALS(SequenceClassificationMixin, _UncertaintyBasedStrategySEALS):
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


class UncertaintyBasedStrategyGradSub(SequenceClassificationMixin, _UncertaintyBasedStrategySEALS):
    def __init__(self, *args, num_neighbours: int = 100, num_influential: int = 10, **kwargs) -> None:
        super().__init__(*args, **kwargs)  # type: ignore
        self.num_neighbours = num_neighbours
        self.to_search = []
        self.pool_subset_ids = []
        self.num_influential = num_influential

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

    def run_query(self, model, datastore: Datastore, query_size: int) -> List[int]:

        train_ids = datastore.get_train_ids()  # type: ignore

        # still no training
        if len(train_ids) == 0:
            pool_loader = self.configure_dataloader(datastore.pool_loader())
            self.progress_tracker.pool_tracker.max = len(pool_loader)  # type: ignore
            return self.compute_most_uncertain(model, pool_loader, query_size)  # type: ignore

        # first loop but we have initial budget
        self.to_search = self.to_search if len(self.to_search) > 0 else train_ids

        # select training instances with the highest gradient norm
        ids_to_search = self.select_train_instances(model.module, datastore)

        # get the embeddings of the labelled instances
        train_embeddings = datastore.get_embeddings(ids_to_search)  # type: ignore

        # get neighbours of training instances from the pool
        nn_ids, _ = datastore.search(  # type: ignore
            query=train_embeddings, query_size=self.num_neighbours, query_in_set=False
        )
        nn_ids = np.unique(np.concatenate(nn_ids).flatten()).tolist()

        return nn_ids[:query_size]

        # self.pool_subset_ids = list(set(nn_ids + self.pool_subset_ids))
        # pool_loader = self.configure_dataloader(datastore.pool_loader(with_indices=self.pool_subset_ids))
        # self.progress_tracker.pool_tracker.max = len(pool_loader)  # type: ignore
        # return self.compute_most_uncertain(model, pool_loader, query_size)  # type: ignore

    def select_train_instances(self, model, datastore) -> List[int]:
        params = {k: v.detach() for k, v in model.named_parameters()}
        buffers = {k: v.detach() for k, v in model.named_buffers()}
        _model = model

        def compute_loss(params: Dict, buffers: Dict, input_ids, attention_mask, labels) -> torch.Tensor:
            inp, att, lab = input_ids.unsqueeze(0), attention_mask.unsqueeze(0), labels.unsqueeze(0)
            return functional_call(_model, (params, buffers), (inp, att), kwargs={"labels": lab}).loss

        def grad_norm(grads, norm_type):
            norms = [g.norm(norm_type).unsqueeze(0) for g in grads.values() if g is not None]
            return torch.concat(norms).norm(norm_type)

        compute_grad = grad(compute_loss)

        def compute_grad_norm(params: Dict, buffers: Dict, input_ids, attention_mask, labels) -> torch.Tensor:
            grads = compute_grad(params, buffers, input_ids, attention_mask, labels)
            return grad_norm(grads, 2)

        compute_grad_norm_vect = vmap(compute_grad_norm, in_dims=(None, None, 0, 0, 0), randomness="same")

        norms, ids = [], []
        for batch in tqdm(datastore.train_loader(), disable=True):
            batch = self.transfer_to_device(batch)
            b_inp, b_att, b_lab = batch["input_ids"], batch["attention_mask"], batch["labels"]
            norms += compute_grad_norm_vect(params, buffers, b_inp, b_att, b_lab).tolist()
            ids += batch[InputKeys.ON_CPU][SpecialKeys.ID]

        norms = np.array(norms)
        ids = np.array(ids)
        topk_ids = norms.argsort()[-self.num_influential :]
        # topk_ids = norms.argsort()[:self.num_influential]
        return ids[topk_ids].tolist()
