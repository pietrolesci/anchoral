import logging
from pathlib import Path

import hydra
from hydra.utils import get_original_cwd, instantiate
from lightning.fabric import seed_everything
from lightning.fabric.loggers import TensorBoardLogger
from omegaconf import DictConfig, OmegaConf
from transformers import AutoModelForSequenceClassification
from transformers.utils.logging import set_verbosity_warning

from energizer.datastores import PandasDataStoreForSequenceClassification

# set logging
set_verbosity_warning()
log = logging.getLogger("hydra")
sep_line = f"{'=' * 70}"


@hydra.main(version_base=None, config_path="../conf", config_name="conf")
def main(cfg: DictConfig) -> None:
    ###############################################################
    # ============ STEP 1: config and initialization ============ #
    ###############################################################

    # resolve interpolation
    OmegaConf.resolve(cfg)
    log.info(f"\n{OmegaConf.to_yaml(cfg)}\n{sep_line}")

    # logging
    log.info(f"Running active learning with strategy {cfg.strategy}")
    if cfg.limit_batches is not None:
        log.critical("!!! DEBUGGING !!!")

    ##################################################
    # ============ STEP 2: data loading ============ #
    ##################################################
    # load data
    data_path = Path(get_original_cwd()) / "data" / "prepared" / cfg.dataset_name
    datastore = PandasDataStoreForSequenceClassification.load(data_path)

    # define initial budget
    if cfg.active_data.budget is not None and cfg.active_data.budget > 0:
        ids = datastore.sample_from_pool(
            size=cfg.active_data.budget, mode="stratified", random_state=cfg.active_data.seed
        )
        datastore.label(
            ids, -1, validation_perc=cfg.active_data.validation_perc, validation_sampling=cfg.active_data.sampling
        )
        log.info(
            f"Initial budget set: labelling {cfg.active_data.budget or 0} samples "
            f"in a {cfg.active_data.sampling} way using seed {cfg.active_data.seed}. "
        )
    log.info(f"Keeping {cfg.active_data.validation_perc} as validation.")

    # set loaders
    datastore.prepare_for_loading(**OmegaConf.to_container(cfg.data))  # type: ignore

    #########################################################
    # ============ STEP 3: model and estimator ============ #
    #########################################################
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

    #####################################################
    # ============ STEP 4: active learning ============ #
    #####################################################
    hparams = {**OmegaConf.to_container(cfg.fit), **OmegaConf.to_container(cfg.active_fit)}  # type: ignore
    fit_out = estimator.active_fit(datastore, **hparams)

    ##################################################
    # ============ STEP 5: save outputs ============ #
    ##################################################
    hparams = {
        **hparams,
        **OmegaConf.to_container(cfg.active_data),  # type: ignore
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
