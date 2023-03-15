import logging
from pathlib import Path

import hydra
from datasets import load_from_disk
from hydra.utils import get_original_cwd, instantiate
from lightning.fabric import seed_everything
from lightning.fabric.loggers import TensorBoardLogger
from omegaconf import DictConfig, OmegaConf
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.data import ClassificationActiveDataModule
from src.energizer.logging import set_ignore_warnings
from src.estimators import resolve_strategy_name

# add resolver for the strategy name
OmegaConf.register_new_resolver("get_name", lambda x: resolve_strategy_name(x))

# set logging
set_ignore_warnings()
log = logging.getLogger("hydra")
sep_line = f"{'=' * 70}"


@hydra.main(version_base=None, config_path="../conf", config_name="conf")
def main(cfg: DictConfig) -> None:
    ###############################################################
    # ============ STEP 1: config and initialization ============ #
    ###############################################################

    # resolve interpolation
    assert cfg.strategy is not None, "You must specify a strategy"
    OmegaConf.resolve(cfg)

    # set paths, load metadata, and tidy config
    data_path = Path(get_original_cwd()) / "data" / "prepared" / cfg.dataset_name

    # load metadata
    metadata = OmegaConf.load(data_path / "metadata.yaml")
    if cfg.model.name_or_path is None:
        cfg.model.name_or_path = metadata.name_or_path

    # toggle balanced loss
    should_load_class_weights = (
        cfg.fit.loss_fn is not None
        and cfg.fit.loss_fn_kwargs is not None
        and "weight" in cfg.fit.loss_fn_kwargs
        and not isinstance(cfg.fit.loss_fn_kwargs.get("weight"), list)
    )

    # logging
    log.info(f"\n{OmegaConf.to_yaml(cfg)}\n{sep_line}")
    log.info(f"Running active learning with strategy {cfg.strategy}")
    if cfg.limit_batches is not None:
        log.critical("!!! DEBUGGING !!!")

    # seed everything
    seed_everything(cfg.seed)
    log.info(f"Seed enabled: {cfg.seed}")

    ##################################################
    # ============ STEP 2: data loading ============ #
    ##################################################
    # load data
    dataset_dict = load_from_disk(data_path, keep_in_memory=True)

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name_or_path)

    # define datamodule
    datamodule = ClassificationActiveDataModule.from_dataset_dict(
        dataset_dict, tokenizer=tokenizer, **OmegaConf.to_container(cfg.data)
    )

    # define initial budget
    if cfg.active_data.budget is not None and cfg.active_data.budget > 0:
        datamodule.set_initial_budget(**OmegaConf.to_container(cfg.active_data))
    log.info(
        f"Initial budget set: labelling {cfg.active_data.budget or 0} samples "
        f"in a {cfg.active_data.sampling} way using seed {cfg.active_data.seed}. "
        f"Keeping {cfg.active_data.validation_perc} as validation."
    )
    log.info(f"Data statistics: {datamodule.data_statistics}")

    # toggle class weights in loss function
    if should_load_class_weights:
        cfg.fit.loss_fn_kwargs = {"weight": datamodule.class_weights}
        log.info(f"Class weights set to: {cfg.fit.loss_fn_kwargs['weight']}")

    ###################################################
    # ============ STEP 3: model loading ============ #
    ###################################################
    # load model using data properties
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model.name_or_path,
        id2label=datamodule.id2label,
        label2id=datamodule.label2id,
        num_labels=len(datamodule.labels),
    )

    ##################################################################
    # ============ STEP 4: define callbacks and loggers ============ #
    ##################################################################
    loggers = instantiate(cfg.loggers) or {}
    callbacks = instantiate(cfg.callbacks) or {}
    log.info(f"Loggers: {loggers}")
    log.info(f"Callbacks: {callbacks}")

    #####################################################
    # ============ STEP 5: active learning ============ #
    #####################################################

    # active learning
    estimator = instantiate(
        cfg.strategy,
        model=model,
        loggers=list(loggers.values()),
        callbacks=list(callbacks.values()),
        **OmegaConf.to_container(cfg.estimator),
        _convert_="all",
    )
    log.info(f"\n{estimator.model_summary}")

    hparams = {
        **OmegaConf.to_container(cfg.fit),
        **OmegaConf.to_container(cfg.active_fit),
        **OmegaConf.to_container(cfg.test),
    }
    # NOTE: `active_fit_end` is overridden to return only test metrics
    fit_out = estimator.active_fit(active_datamodule=datamodule, **hparams)
    log.info(f"Labelled dataset size: {datamodule.train_size}")

    ##################################################
    # ============ STEP 6: save outputs ============ #
    ##################################################

    hparams = {
        **hparams,
        **datamodule.hparams,
        **estimator.hparams,
        **OmegaConf.to_container(cfg.active_data),
        "limit_batches": cfg.limit_batches,
    }
    OmegaConf.save(cfg, "./hparams.yaml")

    # log hparams and test results to tensorboard
    if isinstance(estimator.fabric.logger, TensorBoardLogger):
        estimator.fabric.logger.log_hyperparams(
            params=hparams,
            metrics=fit_out,
        )
    log.info(estimator.progress_tracker)


if __name__ == "__main__":
    main()
