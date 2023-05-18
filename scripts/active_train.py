import logging
from calendar import c

import hydra
from datasets import DatasetDict, load_dataset
from hydra.utils import get_original_cwd, instantiate
from lightning.fabric import seed_everything
from lightning.fabric.loggers import TensorBoardLogger
from omegaconf import DictConfig, OmegaConf
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers.utils.logging import set_verbosity_warning

from energizer.datastores import PandasDataStoreForSequenceClassification
from src.utilities import SEP_LINE, binarize_labels, downsample_positive_class, downsample_test_set, get_initial_budget

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

    ##########################################
    # ============ data loading ============ #
    ##########################################

    # load data
    dataset_dict: DatasetDict = load_dataset(cfg.dataset.absolute_path)  # type: ignore
    dataset_dict.pop("validation", None)  # remove validation

    # maybe binarise label
    positive_class = cfg.dataset.positive_class
    if cfg.dataset.need_binarize is True:
        dataset_dict = binarize_labels(
            dataset_dict=dataset_dict,
            target_name=cfg.dataset.label_column,
            positive_class=cfg.dataset.positive_class,
            logger=log,
        )
        positive_class = 1  # when we binarize we make the positive class equal to 1

    # maybe downsample positive class
    if cfg.dataset.downsample_proportion is not None:
        dataset_dict = downsample_positive_class(
            dataset_dict=dataset_dict,
            target_name=cfg.dataset.label_column,
            positive_class=positive_class,
            proportion=cfg.dataset.downsample_proportion,
            seed=cfg.data.seed,
            logger=log,
        )

    # maybe downsample the test set
    if cfg.dataset.test_set_size is not None:
        dataset_dict = downsample_test_set(
            dataset_dict=dataset_dict,
            target_name=cfg.dataset.label_column,
            positive_class=positive_class,
            test_set_size=cfg.dataset.test_set_size,
            seed=cfg.data.seed,
            logger=log,
        )

    # load tokenizer and tokenize
    tokenizer = AutoTokenizer.from_pretrained(cfg.model)
    dataset_dict = dataset_dict.map(
        lambda ex: tokenizer(ex[cfg.dataset.text_column]), batched=True, desc="Tokenizing", num_proc=8
    )

    ##############################################
    # ============ data preparation ============ #
    ##############################################

    # create datastore for active learning
    datastore = PandasDataStoreForSequenceClassification()
    datastore.from_dataset_dict(
        dataset_dict=dataset_dict,  # type:ignore
        input_names=["input_ids", "attention_mask"],
        target_name=cfg.dataset.label_column,
        uid_name=cfg.dataset.uid_column,
        tokenizer=tokenizer,
    )
    if cfg.embedding_model is not None and "RandomStrategy" not in cfg.strategy._target_:
        emb_col = f"embedding_{cfg.embedding_model}"
        log.info(f"adding index from {emb_col}")
        datastore.add_index(emb_col)

    # define initial budget
    if cfg.active_data.budget is not None and cfg.active_data.budget > 0:
        get_initial_budget(
            datastore=datastore,
            positive_budget=cfg.active_data.positive_budget,
            total_budget=cfg.active_data.budget,
            positive_class=positive_class,
            seed=cfg.active_data.seed,
            validation_perc=cfg.active_data.validation_perc,
            sampling=cfg.active_data.sampling,
            logger=log,
        )

    log.info(f"Keeping {cfg.active_data.validation_perc} as validation.")

    # set loaders
    datastore.prepare_for_loading(**OmegaConf.to_container(cfg.data))  # type: ignore
    log.info(f"Batch:\n{datastore.show_batch('test')}")

    #################################################
    # ============ model and estimator ============ #
    #################################################
    # seed everything
    seed_everything(cfg.seed)
    log.info(f"Seed enabled: {cfg.seed}")

    # load model using data properties
    model = AutoModelForSequenceClassification.from_pretrained(
        datastore.tokenizer.name_or_path,  # type: ignore
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
    estimator.fabric.logger.log_hyperparams(params={**fit_hparams, **OmegaConf.to_container(cfg.active_data)})  # type: ignore

    estimator.active_fit(datastore, **fit_hparams)

    estimator.fabric.logger.finalize("success")


if __name__ == "__main__":
    main()
