from functools import partial
from typing import Dict, List, Tuple, Union

import numpy as np
import torch
from lightning.fabric.wrappers import _FabricModule
from sklearn.utils import check_random_state
from torch import Tensor
from torch.func import functional_call, grad, vmap  # type: ignore
from tqdm.auto import tqdm

from energizer.datastores import PandasDataStoreForSequenceClassification
from energizer.enums import InputKeys, SpecialKeys
from energizer.strategies import RandomStrategy as _RandomStrategy
from energizer.strategies import UncertaintyBasedStrategy
from src.estimators import SequenceClassificationMixin


class RandomStrategy(SequenceClassificationMixin, _RandomStrategy):
    ...


class UncertaintyBasedStrategyPoolSubset(SequenceClassificationMixin, UncertaintyBasedStrategy):
    def __init__(self, *args, num_neighbours: int = 100, seed: int, **kwargs) -> None:
        super().__init__(*args, **kwargs)  # type: ignore
        self.num_neighbours = num_neighbours
        self.pool_subset_ids = []
        self.rng = check_random_state(seed)

        # TODO: compute the influence score of all the training instances at the beginning of training

    def run_query(
        self, model: _FabricModule, datastore: PandasDataStoreForSequenceClassification, query_size: int
    ) -> List[int]:

        # SUBSET
        train_ids = self.get_train_ids(model, datastore)
        # if cold-starting there is no training embedding, fall-back to random sampling
        if len(train_ids) == 0:
            return datastore.sample_from_pool(size=query_size, mode="uniform", random_state=self.rng)

        # TODO: log data that have been used as query and their influence score

        # SEARCH
        # datastore.embedding_name = "embedding_all-mpnet-base-v2"  # FIXME

        train_embeddings = datastore.get_embeddings(train_ids)
        ids, dists = self.search_pool(datastore, train_embeddings, self.num_neighbours)

        # SELECT
        return self.select(model, datastore, ids, dists, query_size)

    def get_train_ids(self, model: _FabricModule, datastore: PandasDataStoreForSequenceClassification) -> List[int]:
        return datastore.get_train_ids()

    def search_pool(
        self, datastore: PandasDataStoreForSequenceClassification, query: np.ndarray, num_neighbours: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get neighbours of training instances from the pool."""
        ids, dists = datastore.search(query=query, query_size=num_neighbours, query_in_set=False)
        ids, dists = np.concatenate(ids).flatten(), np.concatenate(dists).flatten()

        ids, uid_index = np.unique(ids, return_index=True)
        dists = dists[uid_index]
        return ids, dists
        # return np.stack([datastore.sample_from_pool(num_neighbours, "uniform") for _ in range(query.shape[0])]).flatten(), None  # FIXME

    def select(
        self,
        model: _FabricModule,
        datastore: PandasDataStoreForSequenceClassification,
        ids: np.ndarray,
        distances: np.ndarray,
        query_size: int,
    ) -> List[int]:
        pool_loader = self.configure_dataloader(datastore.pool_loader(with_indices=ids.tolist()))
        self.progress_tracker.pool_tracker.max = len(pool_loader)  # type: ignore
        return self.compute_most_uncertain(model, pool_loader, query_size)  # type: ignore


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


class IGALRandom(UncertaintyBasedStrategyPoolSubset):
    def __init__(self, *args, num_influential: int = 10, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.num_influential = num_influential

    def get_train_ids(self, model: _FabricModule, datastore: PandasDataStoreForSequenceClassification) -> List[int]:
        train_ids = datastore.get_train_ids()
        if len(train_ids) > 0:
            return self.rng.choice(train_ids, size=self.num_influential, replace=False).tolist()
        return train_ids


class IGALGradNorm(UncertaintyBasedStrategyPoolSubset):
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

        def _grad_norm(grads: Dict, norm_type: int) -> Tensor:
            norms = [g.norm(norm_type).unsqueeze(0) for g in grads.values() if g is not None]
            return torch.concat(norms).norm(norm_type)

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
        topk_ids = norms.argsort()[-self.num_influential :]  # biggest gradient norm
        return ids[topk_ids].tolist()


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

        def _grad_norm(grads: Dict, norm_type: int) -> Tensor:
            norms = [g.norm(norm_type).unsqueeze(0) for g in grads.values() if g is not None]
            return torch.concat(norms).norm(norm_type)

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
