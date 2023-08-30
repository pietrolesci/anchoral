import logging
import shutil

import hydra
from datasets import DatasetDict, load_from_disk
from hydra.utils import instantiate  # get_original_cwd
from lightning.fabric import seed_everything
from omegaconf import DictConfig, OmegaConf
from tbparse import SummaryReader
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers.utils.logging import set_verbosity_warning

from energizer.active_learning.datastores.classification import ActivePandasDataStoreForSequenceClassification
from energizer.utilities import local_seed
from src.utilities import MODELS, SEP_LINE, get_initial_budget, get_stats_from_dataframe

set_verbosity_warning()
log = logging.getLogger("hydra")


@hydra.main(version_base=None, config_path="../conf", config_name="conf")
def main(cfg: DictConfig) -> None:
    #######################################################
    # ============ config and initialization ============ #
    #######################################################

    # resolve interpolation
    OmegaConf.resolve(cfg)
    log.info(f"\n{OmegaConf.to_yaml(cfg)}\n{SEP_LINE}")
    OmegaConf.save(cfg, "./hparams.yaml")

    # logging
    log.info(f"Running active learning with strategy {cfg.strategy}")
    if cfg.limit_batches is not None:
        log.critical("!!! DEBUGGING !!!")

    seed_everything(cfg.seed)
    log.info(f"Seed enabled: {cfg.seed}")

    ##########################################
    # ============ data loading ============ #
    ##########################################

    # load data
    dataset_dict: DatasetDict = load_from_disk(f"{cfg.dataset.prepared_path}_{cfg.model.name}")  # type: ignore

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODELS[cfg.model.name])

    ##############################################
    # ============ data preparation ============ #
    ##############################################

    # create datastore for active learning
    datastore = ActivePandasDataStoreForSequenceClassification.from_dataset_dict(
        dataset_dict=dataset_dict,  # type:ignore
        input_names=["input_ids", "attention_mask"],
        target_name=cfg.dataset.label_column,
        uid_name=cfg.dataset.uid_column,
        tokenizer=tokenizer,
    )

    # load index
    if "random" not in cfg.strategy["_target_"].lower():
        index_path = f"{cfg.dataset.processed_path}/{cfg.index_metric}"
        log.info(f"loading index from {index_path}")
        index_path, meta_path = f"{index_path}.bin", f"{index_path}.json"
        datastore.load_index(index_path, meta_path)

    # define initial budget
    if cfg.active_data.budget is not None and cfg.active_data.budget > 0:
        ids = get_initial_budget(
            datastore=datastore,
            positive_budget=cfg.active_data.positive_budget,
            total_budget=cfg.active_data.budget,
            positive_class=1,
            seed=cfg.active_data.seed,
        )
        datastore.label(indices=ids, round=-1)
        stats = get_stats_from_dataframe(
            df=datastore.get_by_ids(datastore.get_train_ids()),
            target_name="labels",
            names=["Negative", "Positive"],
        )
        log.info(
            f"Labelled size: {datastore.labelled_size()} "
            f"Pool size: {datastore.pool_size()} "
            f"Test size: {datastore.test_size()}\n"
            f"Label distribution:\n{stats}"
        )

    # set loaders
    datastore.prepare_for_loading(**OmegaConf.to_container(cfg.data))  # type: ignore
    log.info(f"Batch:\n{datastore.show_batch('test')}")

    #################################################
    # ============ model and estimator ============ #
    #################################################

    # load model using data properties
    with local_seed(cfg.model.seed):
        model = AutoModelForSequenceClassification.from_pretrained(
            MODELS[cfg.model.name],  # type: ignore
            id2label=datastore.id2label,
            label2id=datastore.label2id,
            num_labels=len(datastore.labels),
        )

    # define callbacks and loggers
    loggers = instantiate(cfg.loggers) or {}
    callbacks = instantiate(cfg.callbacks) or {}
    log.info(f"Loggers: {loggers}")
    log.info(f"Callbacks: {callbacks}")

    # define estimator
    estimator = instantiate(
        cfg.strategy,
        model=model,
        loggers=list(loggers.values()),
        callbacks=list(callbacks.values()),
        **OmegaConf.to_container(cfg.estimator),  # type: ignore
        _convert_="all",
    )
    log.info(f"\n{estimator.model_summary}")

    #############################################
    # ============ active learning ============ #
    #############################################
    fit_hparams = {**OmegaConf.to_container(cfg.fit), **OmegaConf.to_container(cfg.active_fit)}  # type: ignore
    estimator.logger.log_hyperparams(params={**fit_hparams, **OmegaConf.to_container(cfg.active_data)})  # type: ignore

    estimator.active_fit(datastore, **fit_hparams)

    estimator.logger.finalize("success")
    SummaryReader(str(estimator.logger.log_dir)).scalars.to_parquet("tb_logs.parquet")
    shutil.rmtree(".model_cache")


if __name__ == "__main__":
    main()
