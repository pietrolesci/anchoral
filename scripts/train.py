import logging

import hydra
from datasets import DatasetDict, load_dataset
from hydra.utils import instantiate
from lightning.fabric import seed_everything
from omegaconf import DictConfig, OmegaConf
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers.utils.logging import set_verbosity_warning

from energizer.datastores import PandasDataStoreForSequenceClassification
from src.estimators import EstimatorForSequenceClassification
from src.utilities import SEP_LINE, binarize_labels, downsample_positive_class, downsample_test_set

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
    estimator = EstimatorForSequenceClassification(
        model=model,
        **OmegaConf.to_container(cfg.estimator),  # type: ignore
        loggers=list(loggers.values()),
        callbacks=list(callbacks.values()),
    )
    log.info(f"\n{estimator.model_summary}")

    ######################################
    # ============ training ============ #
    ######################################
    fit_out = estimator.fit(
        train_loader=datastore.train_loader(passive=True),
        **OmegaConf.to_container(cfg.fit),  # type: ignore
    )

    # test: in our src.huggingface.estimators.EstimatorForSequenceClassification we overwritten
    # the test_epoch_end method and we return a dictionary with the metrics, so test_out.output
    # is a Dict
    test_out = estimator.test(datastore.test_loader(), **OmegaConf.to_container(cfg.test))  # type: ignore

    estimator.fabric.logger.log_hyperparams(
        params={
            **OmegaConf.to_container(cfg.data),  # type: ignore
            **OmegaConf.to_container(cfg.fit),  # type: ignore
            **OmegaConf.to_container(cfg.test),  # type: ignore
        }
    )  # type: ignore
    estimator.fabric.logger.finalize("success")


if __name__ == "__main__":
    main()
