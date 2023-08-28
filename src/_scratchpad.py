import math
from functools import partial
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from lightning.fabric.wrappers import _FabricModule
from matplotlib.pyplot import annotate
from scipy.special import softmax
from sklearn.utils import check_random_state
from torch import Tensor
from torch.func import functional_call, grad, vmap  # type: ignore
from tqdm.auto import tqdm

from energizer.datastores import PandasDataStoreForSequenceClassification
from energizer.enums import InputKeys, SpecialKeys
from energizer.strategies import RandomStrategy as _RandomStrategy
from energizer.strategies import UncertaintyBasedStrategy as _UncertaintyBasedStrategy
from src.estimator import SequenceClassificationMixin


class PoolSamplingMixin:
    def get_pool_ids(
        self,
        ids: np.ndarray,
        distances: np.ndarray,
        train_ids: List[int],
        datastore: PandasDataStoreForSequenceClassification,
    ) -> np.ndarray:

        new_df = pd.DataFrame(
            {
                SpecialKeys.ID: ids.flatten(),
                "dists": distances.flatten(),
                "train_uid": np.repeat(train_ids, ids.shape[1], axis=0).flatten(),
            }
        )

        # add the label to current_pool in order to sample by class
        new_df = pd.merge(
            new_df,
            datastore.data.loc[datastore._train_mask(), [SpecialKeys.ID, "labels"]],
            left_on="train_uid",
            right_on="uid",
            suffixes=["", "_drop"],
            how="left",
        )
        new_df = new_df.drop(columns=[f"{SpecialKeys.ID}_drop"])
        new_df = (
            new_df.groupby([SpecialKeys.ID, "labels"])
            .agg(dists=("dists", "mean"), train_uid=("train_uid", "unique"))
            .reset_index()
        )
        # mean works so far
        # assert len(self.temperatures) == new_df["labels"].nunique()

        # since for cosine distances lowest is better, change it to higher is better
        new_df["scores"] = 1 - new_df["dists"]

        samples = []
        # sample positive class
        pos_df = new_df.loc[new_df["labels"] == 1]
        if len(pos_df) > 0:
            probs = self.get_probs(pos_df["scores"].values, 1)  # type: ignore
            size = min(int(self.subset_size * 3 / 4), len(pos_df))  # type: ignore
            samples += self.rng.choice(pos_df[SpecialKeys.ID].values, size=size, replace=False, p=probs).tolist()  # type: ignore

        # sample negative class
        neg_df = new_df.loc[(new_df["labels"] == 0) & (~new_df[SpecialKeys.ID].isin(samples))]
        if len(neg_df) > 0:
            probs = self.get_probs(neg_df["scores"].values, 0)  # type: ignore
            size = min(self.subset_size - len(samples), len(neg_df))  # type: ignore
            samples += self.rng.choice(neg_df[SpecialKeys.ID].values, size=size, replace=False, p=probs).tolist()  # type: ignore

        if len(samples) < self.subset_size:  # type: ignore
            n = self.subset_size - len(samples)  # type: ignore
            samples += datastore.sample_from_pool(size=n, mode="uniform", random_state=self.rng)  # type: ignore

        return np.array(samples)

    def get_probs(self, scores: np.ndarray, class_idx: int) -> np.ndarray:
        temp = self.temperatures[class_idx]  # type: ignore
        scores = scores / temp
        return softmax(scores, axis=0)


class RandomGuideWithSampling(PoolSamplingMixin, RandomGuide):
    def __init__(self, *args, temperatures: List[float], **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.temperatures = temperatures


class FullGuideWithSampling(PoolSamplingMixin, FullGuide):
    def __init__(self, *args, temperatures: List[float], **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.temperatures = temperatures


def get_pool_ids(
    self,
    ids: np.ndarray,
    distances: np.ndarray,
    train_ids: List[int],
    datastore: PandasDataStoreForSequenceClassification,
) -> np.ndarray:

    # add the distances to the current_pool and when the same uid is picked by several train_uids,
    # just keep the one with the lowest distance. This call updates current_pool
    _ = super().get_pool_ids(ids, distances, train_ids, datastore)

    # add the label to current_pool in order to sample by class
    new_df = pd.merge(
        self.current_pool,
        datastore.data.loc[datastore._train_mask(), [SpecialKeys.ID, "labels"]],
        left_on="train_uid",
        right_on="uid",
        suffixes=["", "_drop"],
        how="left",
    )
    new_df = new_df.drop(columns=[f"{SpecialKeys.ID}_drop"])
    # assert len(self.temperatures) == new_df["labels"].nunique()

    # since for cosine distances lowest is better, change it to higher is better
    new_df["scores"] = 1 - new_df["dists"]

    # sample per class
    samples = []
    for c, df in new_df.groupby("labels"):
        if c == 0:
            continue
        probs = self.get_probs(df["scores"].values, c)  # type: ignore
        # size = min(int(self.subset_size / 2), len(df))
        size = min(int(self.subset_size), len(df))
        samples += self.rng.choice(df[SpecialKeys.ID].values, size=size, replace=False, p=probs).tolist()  # type: ignore

    return np.array(samples)


def get_probs(self, scores: np.ndarray, class_idx: int) -> np.ndarray:
    temp = self.temperatures[class_idx]
    scores = scores / temp
    return softmax(scores, axis=0)


class GradNormGuide(FullGuide):
    def __init__(self, *args, num_influential: int, norm_type: int, **kwargs) -> None:
        super().__init__(*args, **kwargs)  # type: ignore
        self.num_influential = num_influential
        self.norm_type = norm_type

    def get_train_ids(self, model: _FabricModule, datastore: PandasDataStoreForSequenceClassification) -> List[int]:
        _model = model.module
        params = {k: v.detach() for k, v in _model.named_parameters()}
        buffers = {k: v.detach() for k, v in _model.named_buffers()}

        def _grad_norm(grads: Dict, norm_type: int) -> Tensor:
            """This is a good way of computing the norm for all parameters across the network.

            Check [here](https://github.com/viking-sudo-rm/norm-growth/blob/bca0576242c21de0ee06cdc3561dd27aa88a7040/finetune_trans.py#L89)
            for confirmation: they return the same results but this implementation is more convenient
            because we apply the norm layer-wise first and then we further aggregate.
            """
            norms = [g.norm(norm_type).unsqueeze(0) for g in grads.values() if g is not None]
            return torch.concat(norms).norm(norm_type)

        def compute_loss(
            params: Dict, buffers: Dict, input_ids: Tensor, attention_mask: Tensor, labels: Tensor
        ) -> torch.Tensor:
            inp, att, lab = input_ids.unsqueeze(0), attention_mask.unsqueeze(0), labels.unsqueeze(0)
            return functional_call(_model, (params, buffers), (inp, att), kwargs={"labels": lab}).loss

        grad_norm = partial(_grad_norm, norm_type=self.norm_type)

        compute_grad = grad(compute_loss)

        def compute_grad_norm(
            params: Dict, buffers: Dict, input_ids: Tensor, attention_mask: Tensor, labels: Tensor
        ) -> torch.Tensor:
            grads = compute_grad(params, buffers, input_ids, attention_mask, labels)
            return grad_norm(grads)

        compute_grad_norm_vect = vmap(compute_grad_norm, in_dims=(None, None, 0, 0, 0), randomness="same")

        norms, ids = [], []
        for batch in tqdm(datastore.train_loader(), disable=True):
            batch = self.transfer_to_device(batch)
            b_inp, b_att, b_lab = batch[InputKeys.INPUT_IDS], batch[InputKeys.ATT_MASK], batch[InputKeys.TARGET]
            norms += compute_grad_norm_vect(params, buffers, b_inp, b_att, b_lab).tolist()
            ids += batch[InputKeys.ON_CPU][SpecialKeys.ID]

        norms = np.array(norms)
        ids = np.array(ids)
        topk_ids = ids[norms.argsort()[-self.num_influential :]]  # biggest gradient norm
        return topk_ids.tolist()


class GradNormGuideWithSampling(PoolSamplingMixin, GradNormGuide):
    def __init__(self, *args, temperatures: List[float], **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.temperatures = temperatures

    # def search_pool(
    #     self, datastore: PandasDataStoreForSequenceClassification, query: np.ndarray
    # ) -> Tuple[np.ndarray, np.ndarray]:
    #     """Get neighbours of training instances from the pool."""
    #     num_neighbours = max(self.num_neighbours, int(self.subset_size / query.shape[0])) * 10
    #     return datastore.search(query=query, query_size=num_neighbours, query_in_set=False)

    # def get_pool_ids(
    #     self,
    #     ids: np.ndarray,
    #     distances: np.ndarray,
    #     train_ids: List[int],
    #     datastore: PandasDataStoreForSequenceClassification,
    # ) -> np.ndarray:

    #     new_df = pd.DataFrame(
    #         {
    #             SpecialKeys.ID: ids.flatten(),
    #             "dists": distances.flatten(),
    #             "train_uid": np.repeat(train_ids, ids.shape[1], axis=0).flatten(),
    #         }
    #     )

    #     # # add the distances to the current_pool and when the same uid is picked by several train_uids,
    #     # # just keep the one with the lowest distance. This call updates current_pool
    #     # _ = super().get_pool_ids(ids, distances, train_ids, datastore)

    #     # add the label to current_pool in order to sample by class
    #     new_df = pd.merge(
    #         new_df,
    #         datastore.data.loc[datastore._train_mask(), [SpecialKeys.ID, "labels"]],
    #         left_on="train_uid",
    #         right_on="uid",
    #         suffixes=["", "_drop"],
    #         how="left",
    #     )
    #     new_df = new_df.drop(columns=[f"{SpecialKeys.ID}_drop"])
    #     new_df = (
    #         new_df.groupby([SpecialKeys.ID, "labels"])
    #         .agg(dists=("dists", "mean"), train_uid=("train_uid", "unique"))
    #         .reset_index()
    #     )
    #     # mean works so far
    #     # assert len(self.temperatures) == new_df["labels"].nunique()

    #     # since for cosine distances lowest is better, change it to higher is better
    #     new_df["scores"] = 1 - new_df["dists"]

    #     samples = []
    #     # sample positive class
    #     pos_df = new_df.loc[new_df["labels"] == 1]
    #     if len(pos_df) > 0:
    #         probs = self.get_probs(pos_df["scores"].values, 1)  # type: ignore
    #         size = min(int(self.subset_size * 3/4), len(pos_df))
    #         samples += self.rng.choice(pos_df[SpecialKeys.ID].values, size=size, replace=False, p=probs).tolist()  # type: ignore

    #     # sample negative class
    #     neg_df = new_df.loc[(new_df["labels"] == 0) & (~new_df[SpecialKeys.ID].isin(samples))]
    #     if len(neg_df) > 0:
    #         probs = self.get_probs(neg_df["scores"].values, 0)  # type: ignore
    #         size = min(self.subset_size - len(samples), len(neg_df))
    #         samples += self.rng.choice(neg_df[SpecialKeys.ID].values, size=size, replace=False, p=probs).tolist()  # type: ignore

    #     if len(samples) < self.subset_size:
    #         n = self.subset_size - len(samples)
    #         samples += datastore.sample_from_pool(size=n, mode="uniform", random_state=self.rng)

    #     return np.array(samples)

    # def get_probs(self, scores: np.ndarray, class_idx: int) -> np.ndarray:
    #     temp = self.temperatures[class_idx]
    #     scores = scores / temp
    #     return softmax(scores, axis=0)


class IGAL(UncertaintyBasedStrategyPoolSubset):
    def __init__(self, *args, num_influential: int = 10, norm_type: int = 2, **kwargs) -> None:
        super().__init__(*args, **kwargs)  # type: ignore
        self.num_influential = num_influential
        self.norm_type = norm_type

    def get_train_ids(self, model: _FabricModule, datastore: PandasDataStoreForSequenceClassification) -> List[int]:
        _model = model.module
        params = {k: v.detach() for k, v in _model.named_parameters()}
        buffers = {k: v.detach() for k, v in _model.named_buffers()}

        def compute_loss(
            params: Dict, buffers: Dict, input_ids: Tensor, attention_mask: Tensor, labels: Tensor
        ) -> torch.Tensor:
            inp, att, lab = input_ids.unsqueeze(0), attention_mask.unsqueeze(0), labels.unsqueeze(0)
            return functional_call(_model, (params, buffers), (inp, att), kwargs={"labels": lab}).loss

        grad_norm = partial(_grad_norm, norm_type=self.norm_type)

        compute_grad = grad(compute_loss)

        def compute_hessian() -> Dict:
            ...

        def hvp(hessian: Dict, grads: Dict) -> Dict:
            ...

        hessian = compute_hessian()

        def compute_grad_norm(
            params: Dict, buffers: Dict, input_ids: Tensor, attention_mask: Tensor, labels: Tensor
        ) -> torch.Tensor:
            grads = compute_grad(params, buffers, input_ids, attention_mask, labels)
            grads = hvp(hessian, grads)
            return grad_norm(grads)

        compute_grad_norm_vect = vmap(compute_grad_norm, in_dims=(None, None, 0, 0, 0), randomness="same")

        norms, ids = [], []
        for batch in tqdm(datastore.train_loader(), disable=True):
            batch = self.transfer_to_device(batch)
            b_inp, b_att, b_lab = batch[InputKeys.INPUT_IDS], batch[InputKeys.ATT_MASK], batch[InputKeys.TARGET]
            norms += compute_grad_norm_vect(params, buffers, b_inp, b_att, b_lab).tolist()
            ids += batch[InputKeys.ON_CPU][SpecialKeys.ID]

        norms = np.array(norms)
        ids = np.array(ids)
        topk_ids = norms.argsort()[-self.num_influential :]  # biggest gradient norm
        return ids[topk_ids].tolist()


class SEALS(UncertaintyBasedStrategyPoolSubset):
    pool_subset_ids: List[int] = []
    to_search: List[int] = []

    def get_train_ids(self, model: _FabricModule, datastore: PandasDataStoreForSequenceClassification) -> List[int]:
        # we will only search the currently added train_ids sinces the others are cached into pool_subset_ids
        if len(self.to_search) == 0:
            self.to_search = datastore.get_train_ids()
        return self.to_search

    def select(
        self,
        model: _FabricModule,
        datastore: PandasDataStoreForSequenceClassification,
        ids: np.ndarray,
        distances: np.ndarray,
        query_size: int,
    ) -> List[int]:
        # add the NNs of the current train_ids to the pool
        self.pool_subset_ids = list(set(self.pool_subset_ids + ids.tolist()))

        # select the ids that need to be labelled
        pool_loader = self.configure_dataloader(datastore.pool_loader(with_indices=self.pool_subset_ids))
        self.progress_tracker.pool_tracker.max = len(pool_loader)  # type: ignore
        annotated_ids = self.compute_most_uncertain(model, pool_loader, query_size)  # type: ignore

        # remove those ids that are annotated
        self.pool_subset_ids = list(set(self.pool_subset_ids).difference(set(annotated_ids)))

        # overwrite with the train_ids that need to be searched
        self.to_search = annotated_ids

        return annotated_ids
