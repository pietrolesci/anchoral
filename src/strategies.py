from functools import partial
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from lightning.fabric.wrappers import _FabricModule
from sklearn.utils import check_random_state
from torch import Tensor
from torch.func import functional_call, grad, vmap  # type: ignore
from tqdm.auto import tqdm
import pandas as pd
from energizer.datastores import PandasDataStoreForSequenceClassification
from energizer.enums import InputKeys, SpecialKeys
from energizer.strategies import RandomStrategy as _RandomStrategy
from energizer.strategies import UncertaintyBasedStrategy
from src.estimators import SequenceClassificationMixin


class RandomStrategy(SequenceClassificationMixin, _RandomStrategy):
    ...


class UncertaintyBasedStrategyPoolSubset(SequenceClassificationMixin, UncertaintyBasedStrategy):

    def __init__(self, *args, num_neighbours: int, subset_size: int, seed: int, **kwargs) -> None:
        super().__init__(*args, **kwargs)  # type: ignore
        self.num_neighbours = num_neighbours
        self.subset_size = subset_size
        self.rng = check_random_state(seed)
        self.pool_subset_ids = []

        # TODO: compute the influence score of all the training instances at the beginning of training

    def run_query(
        self, model: _FabricModule, datastore: PandasDataStoreForSequenceClassification, query_size: int
    ) -> List[int]:
        # TODO: log data that have been used as query and their influence score

        train_ids = self.get_train_ids(model, datastore)
        
        if len(train_ids) == 0:
            # if cold-starting there is no training embedding, fall-back to random sampling
            return datastore.sample_from_pool(size=query_size, mode="uniform", random_state=self.rng)

        # SEARCH
        train_embeddings = datastore.get_embeddings(train_ids)
        ids, dists = self.search_pool(datastore, train_embeddings, self.num_neighbours)

        # SUBSET
        ids = self.get_pool_ids(ids, dists, train_ids)

        # SELECT
        return self.select(model, datastore, ids, dists, query_size)

    def get_train_ids(self, model: _FabricModule, datastore: PandasDataStoreForSequenceClassification) -> List[int]:
        """Get all training ids by default."""
        return datastore.get_train_ids()

    def search_pool(
        self, datastore: PandasDataStoreForSequenceClassification, query: np.ndarray, num_neighbours: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get neighbours of training instances from the pool."""
        return datastore.search(query=query, query_size=num_neighbours, query_in_set=False)

    def get_pool_ids(self, ids: np.ndarray, distances: np.ndarray, train_ids: List[int]) -> np.ndarray:
        """Given all the matches get the subset_size closest."""
        # order the ids by their distance from a train datapoint (smaller is more similar)
        ids = ids.flatten()[distances.flatten().argsort()]
        # deduplicate keeping the order of first appearance
        _, udx = np.unique(ids, return_index=True)
        # return the ordered set
        return ids[np.sort(udx)][:self.subset_size]

    def select(
        self,
        model: _FabricModule,
        datastore: PandasDataStoreForSequenceClassification,
        ids: np.ndarray,
        distances: Optional[np.ndarray],
        query_size: int,
    ) -> List[int]:
        pool_loader = self.configure_dataloader(datastore.pool_loader(with_indices=ids.tolist()))
        self.progress_tracker.pool_tracker.max = len(pool_loader)  # type: ignore
        return self.compute_most_uncertain(model, pool_loader, query_size)  # type: ignore


class RandomSubset(UncertaintyBasedStrategyPoolSubset):
    def run_query(
        self, model: _FabricModule, datastore: PandasDataStoreForSequenceClassification, query_size: int
    ) -> List[int]:
        subset_size = min(datastore.pool_size(), self.subset_size)
        ids = datastore.sample_from_pool(size=subset_size, mode="uniform", random_state=self.rng)
        return self.select(model, datastore, np.array(ids), None, query_size)


class FullSubset(UncertaintyBasedStrategyPoolSubset):
    current_pool: pd.DataFrame = pd.DataFrame(columns=["train_ids", "ids", "dists"])
    to_search: List[int] = []

    def get_train_ids(self, model: _FabricModule, datastore: PandasDataStoreForSequenceClassification) -> List[int]:
        # we will only search the currently added train_ids sinces the others are cached into pool_subset_ids
        if len(self.to_search) == 0:
            # in the first round search everything
            self.to_search = datastore.get_train_ids()
        # then only the newly added training points
        return self.to_search
    
    def get_pool_ids(self, ids: np.ndarray, distances: np.ndarray, train_ids: List[int]) -> np.ndarray:
        new_df = pd.DataFrame({
            "ids": ids.flatten(), 
            "dists": distances.flatten(), 
            "train_ids": np.repeat(train_ids, ids.shape[1], axis=0).flatten(),
        })        
        df = pd.concat([self.current_pool, new_df], axis=0, ignore_index=False)
        
        # sort by distance (lower first) and then deduplicate keeping the first instance
        df = df.sort_values("dists", ascending=True).drop_duplicates(subset=["ids"], keep="first")
        
        self.current_pool = df.iloc[:self.subset_size]
        
        return self.current_pool["ids"].to_numpy()


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
        super().__init__(*args, **kwargs)  # type: ignore
        self.num_influential = num_influential

    def get_train_ids(self, model: _FabricModule, datastore: PandasDataStoreForSequenceClassification) -> List[int]:
        train_ids = datastore.get_train_ids()
        if len(train_ids) > 0:
            return self.rng.choice(train_ids, size=min(self.num_influential, len(train_ids)), replace=False).tolist()
        return train_ids


def _grad_norm(grads: Dict, norm_type: int) -> Tensor:
    """This is a good way of computing the norm for all parameters across the network.

    Check [here](https://github.com/viking-sudo-rm/norm-growth/blob/bca0576242c21de0ee06cdc3561dd27aa88a7040/finetune_trans.py#L89)
    for confirmation: they return the same results but this implementation is more convenient
    because we apply the norm layer-wise first and then we further aggregate.
    """
    norms = [g.norm(norm_type).unsqueeze(0) for g in grads.values() if g is not None]
    return torch.concat(norms).norm(norm_type)


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
