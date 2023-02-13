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

from src.huggingface.datamodule import ClassificationDataModule
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

    # seed everything
    seed_everything(cfg.seed)

    # ============ STEP 2: data loading ============
    # load data
    dataset_dict = load_from_disk(data_path, keep_in_memory=True)

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name_or_path)

    # define datamodule
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

    # define loggers and callbacks
    loggers = instantiate(cfg.loggers) or {}
    callbacks = instantiate(cfg.callbacks) or {}
    log.info(f"Loggers: {loggers}")
    log.info(f"Callbacks: {callbacks}")

    # define estimator
    estimator = EstimatorForSequenceClassification(
        model=model,
        **OmegaConf.to_container(cfg.trainer),
        loggers=list(loggers.values()),
        callbacks=list(callbacks.values()),
    )

    # fit
    fit_out = estimator.fit(
        train_loader=datamodule.train_loader(),
        validation_loader=datamodule.validation_loader(),
        **OmegaConf.to_container(cfg.fit),
    )

    # test
    test_out = estimator.test(datamodule.test_loader(), **OmegaConf.to_container(cfg.test))

    # ============ STEP 4: save outputs ============
    OmegaConf.save(cfg, "./hparams.yaml")

    # log hparams to tensorboard
    if isinstance(estimator.fabric.logger, TensorBoardLogger):
        hparams = OmegaConf.to_container(cfg)
        metrics = srsly.read_json(f"{cfg.callbacks.save_outputs.dirpath}/test/epoch_level.json")
        hparam_dict = {**hparams["trainer"], **hparams["model"], **hparams["data"], **hparams["fit"]}
        metric_dict = {f"test_{k}": v for k, v in metrics["metrics"].items()}
        metric_dict["test_loss"] = metrics["loss"]
        estimator.fabric.logger.experiment.add_hparams(
            hparam_dict=hparam_dict,
            metric_dict=metric_dict,
        )

    # # save experiment output
    # with open("./train_outputs", "wb") as fl:
    #     pickle.dump(train_outputs, fl)

    # with open("./test_outputs", "wb") as fl:
    #     pickle.dump(test_outputs, fl)


if __name__ == "__main__":
    main()
