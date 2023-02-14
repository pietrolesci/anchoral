import logging
from pathlib import Path

import hydra
import srsly
from datasets import load_from_disk
from hydra.utils import get_original_cwd, instantiate
from lightning.fabric import seed_everything
from lightning.fabric.loggers import TensorBoardLogger
from omegaconf import DictConfig, OmegaConf
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.huggingface.datamodule import ClassificationActiveDataModule, ClassificationDataModule
from src.huggingface.estimators import EstimatorForSequenceClassification
from src.logging import set_ignore_warnings

set_ignore_warnings()
log = logging.getLogger("hydra")
sep_line = f"{'=' * 70}"


@hydra.main(version_base=None, config_path="../conf", config_name="conf")
def main(cfg: DictConfig) -> None:
    # ============ STEP 1: config and initialization ============
    # resolve interpolation
    OmegaConf.resolve(cfg)

    # set paths, load metadata, and tidy config
    data_path = Path(get_original_cwd()) / "data" / "prepared" / cfg.dataset_name

    metadata = OmegaConf.load(data_path / "metadata.yaml")
    if cfg.model.name_or_path is None:
        cfg.model.name_or_path = metadata.name_or_path

    log.info(f"\n{OmegaConf.to_yaml(cfg)}\n{sep_line}")
    if cfg.dry_run:
        log.critical("\n\n\t !!! DEBUGGING !!! \n\n")

    # toggle balanced loss
    should_load_class_weights = (
        cfg.fit.loss_fn is not None
        and cfg.fit.loss_fn_kwargs is not None
        and "weight" in cfg.fit.loss_fn_kwargs
        and not isinstance(cfg.fit.loss_fn_kwargs.get("weight"), list)
    )

    should_run_active_learning = cfg.strategy is not None

    # seed everything
    seed_everything(cfg.seed)

    # ============ STEP 2: data loading ============
    # load data
    dataset_dict = load_from_disk(data_path, keep_in_memory=True)

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name_or_path)

    # define datamodule
    if should_run_active_learning:
        datamodule = ClassificationActiveDataModule.from_dataset_dict(
            dataset_dict, tokenizer=tokenizer, **OmegaConf.to_container(cfg.data)
        )
    else:
        datamodule = ClassificationDataModule.from_dataset_dict(
            dataset_dict, tokenizer=tokenizer, **OmegaConf.to_container(cfg.data)
        )

    if should_load_class_weights:
        cfg.fit.loss_fn_kwargs = {"weight": datamodule.class_weights}
        log.info(f"Class weights set to: {cfg.fit.loss_fn_kwargs['weight']}")

    # ============ STEP 3: model loading ============
    # load model using data properties
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model.name_or_path,
        id2label=datamodule.id2label,
        label2id=datamodule.label2id,
        num_labels=len(datamodule.labels),
    )

    # ============ STEP 4: define callbacks and loggers ============
    loggers = instantiate(cfg.loggers) or {}
    callbacks = instantiate(cfg.callbacks) or {}
    log.info(f"Loggers: {loggers}")
    log.info(f"Callbacks: {callbacks}")

    # ============ STEP 5: toggle normal training and active learning ============
    if should_run_active_learning:
        # active learning
        active_estimator = instantiate(
            cfg.strategy,
            model=model,
            loggers=list(loggers.values()),
            callbacks=list(callbacks.values()),
            **OmegaConf.to_container(cfg.estimator),
            _convert_="all",
        )

        active_fit_out = active_estimator.active_fit(
            active_datamodule=datamodule, **OmegaConf.to_container(cfg.active_fit), **OmegaConf.to_container(cfg.fit)
        )

        # save labelled dataset
        datamodule.save_labelled_dataset(".")

    else:
        # proceed with the normal training

        # define estimator
        estimator = EstimatorForSequenceClassification(
            model=model,
            **OmegaConf.to_container(cfg.estimator),
            loggers=list(loggers.values()),
            callbacks=list(callbacks.values()),
        )

        # fit
        fit_out = estimator.fit(
            train_loader=datamodule.train_loader(),
            validation_loader=datamodule.validation_loader(),
            **OmegaConf.to_container(cfg.fit),
        )

        # in our src.huggingface.estimators.EstimatorForSequenceClassification we have overwritten
        # the test_epoch_end method and we return a dictionary with the metrics, so test_out.output
        # is a Dict
        test_out = estimator.test(datamodule.test_loader(), **OmegaConf.to_container(cfg.test))

        # log hparams and test results to tensorboard
        if isinstance(estimator.fabric.logger, TensorBoardLogger):
            estimator.fabric.logger.experiment.add_hparams(
                hparam_dict={**datamodule.hparams, **estimator.hparams, **fit_out.hparams},
                metric_dict=test_out.output,
            )

    # ============ STEP 6: save hparams ============
    OmegaConf.save(cfg, "./hparams.yaml")


if __name__ == "__main__":
    main()
