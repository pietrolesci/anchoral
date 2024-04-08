import logging

import hydra
from datasets import DatasetDict, load_from_disk
from hydra.utils import instantiate  # get_original_cwd
from lightning.fabric import seed_everything
from omegaconf import DictConfig, OmegaConf
from pandas import DataFrame
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers.utils.logging import set_verbosity_warning

from energizer.active_learning.datastores.classification import ActivePandasDataStoreForSequenceClassification
from energizer.utilities import local_seed
from src.strategies import STRATEGIES
from src.utilities import MODELS, SEP_LINE, get_initial_budget, get_stats_from_dataframe, remove_dir

set_verbosity_warning()
log = logging.getLogger("hydra")


def parse_hparams(cfg: DictConfig) -> tuple[dict, dict]:
    def parse_and_add_suffix(conf: DictConfig, prefix: str) -> dict:
        return {k if k != "seed" else f"{prefix}_{k}": v for k, v in OmegaConf.to_container(conf).items()}  # type: ignore

    fit_hparams = {**OmegaConf.to_container(cfg.fit), **OmegaConf.to_container(cfg.active_fit)}  # type: ignore
    all_hparams = {
        "global_seed": cfg.seed,
        "index_metric": cfg.index_metric,
        "strategy": cfg.strategy.name,
        "dataset": cfg.dataset.name,
        **fit_hparams,
        **parse_and_add_suffix(cfg.active_data, "initial_set"),  # type: ignore
        **parse_and_add_suffix(cfg.data, "data_order"),  # type: ignore
        **parse_and_add_suffix(cfg.model, "model_init"),  # type: ignore
        **parse_and_add_suffix(cfg.estimator, "estimator"),  # type: ignore
        **parse_and_add_suffix(cfg.strategy.args, "strategy"),  # type: ignore
    }
    return fit_hparams, all_hparams


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
    dataset_dict: DatasetDict = load_from_disk(f"{cfg.dataset.prepared_path}_{cfg.model.name}", keep_in_memory=True)  # type: ignore

    # load tokenizer
    if cfg.model.name == "t5-base":
        tokenizer = AutoTokenizer.from_pretrained(MODELS[cfg.model.name], legacy=False)
    else:
        tokenizer = AutoTokenizer.from_pretrained(MODELS[cfg.model.name])

    if cfg.model.name == "gpt2":
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    ##############################################
    # ============ data preparation ============ #
    ##############################################

    # create datastore for active learning
    datastore = ActivePandasDataStoreForSequenceClassification.from_dataset_dict(
        dataset_dict=dataset_dict,  # type:ignore
        uid_name=cfg.dataset.uid_column,
        tokenizer=tokenizer,
    )

    # load index
    if "random" not in cfg.strategy["name"]:
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
            minority_classes=cfg.dataset["minority_classes"],
            seed=cfg.active_data.seed,
        )
        datastore.label(indices=ids, round=-1)
        stats = get_stats_from_dataframe(
            df=datastore.get_by_ids(datastore.get_train_ids()), target_name="labels", names=datastore.labels
        )
        log.info(
            f"Labelled size: {datastore.labelled_size()} "
            f"Pool size: {datastore.pool_size()} "
            f"Test size: {datastore.test_size()}\n"
            f"Label distribution:\n{DataFrame(stats).to_markdown()}"
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
        if cfg.model.name == "gpt2":
            model.config.pad_token_id = model.config.eos_token_id
        elif cfg.model.name == "t5-base":
            model.num_labels = len(datastore.labels)

    # define callbacks and loggers
    loggers = instantiate(cfg.loggers) or {}
    callbacks = instantiate(cfg.callbacks) or {}
    log.info(f"Loggers: {loggers}")
    log.info(f"Callbacks: {callbacks}")

    # define estimator
    estimator = STRATEGIES[cfg.strategy.name](
        model=model,
        loggers=list(loggers.values()),
        callbacks=list(callbacks.values()),
        **OmegaConf.to_container(cfg.estimator),  # type: ignore
        **OmegaConf.to_container(cfg.strategy.args),  # type: ignore
    )
    # this is a hack -- otherwise I would need to change the definition of all strategies
    # instead I just set this in src.estimator.SequenceClassificationMixin
    estimator.minority_class_ids = cfg.dataset["minority_classes"]

    log.info(f"\n{estimator.model_summary}")

    #############################################
    # ============ active learning ============ #
    #############################################

    # parse hparams
    fit_hparams, all_hparams = parse_hparams(cfg)

    # log hyper-parameters
    for logger in estimator.loggers:
        logger.log_hyperparams(params=all_hparams)

    # run
    estimator.active_fit(datastore, **fit_hparams)

    # finalise logging
    for logger in estimator.loggers:
        logger.save_to_parquet(f"{logger.logger_name}_logs.parquet")
        logger.finalize("success")

    for path in (".model_cache", ".checkpoints", "wandb"):
        remove_dir(path)


if __name__ == "__main__":
    main()
