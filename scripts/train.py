import logging
import pickle
from pathlib import Path

import hydra
import srsly
from datasets import load_from_disk
from hydra.utils import get_original_cwd
from lightning.fabric import seed_everything
from omegaconf import DictConfig, OmegaConf
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.data.datamodule import ClassificationDataModule
from src.estimators.classification import EstimatorForSequenceClassification
from src.logging import set_ignore_warnings

set_ignore_warnings()
# log = get_logger("hydra")
log = logging.getLogger("train")
sep_line = f"{'=' * 70}"


@hydra.main(version_base=None, config_path="../conf", config_name="conf")
def main(cfg: DictConfig):

    log.info(f"\n{OmegaConf.to_yaml(cfg, resolve=True)}\n{sep_line}")
    if cfg.dry_run:
        log.critical("\n\n\t !!! DEBUGGING !!! \n\n")

    # seed everything
    seed_everything(cfg.seed)

    # load data and metadata
    data_path = Path(get_original_cwd()) / "data" / "prepared" / cfg.dataset_name
    metadata = OmegaConf.load(data_path / "metadata.yaml")
    dataset_dict = load_from_disk(data_path, keep_in_memory=True)

    # load tokenizer
    if cfg.model.name_or_path is None:
        cfg.model.name_or_path = metadata.name_or_path
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name_or_path)

    # define datamodule
    datamodule = ClassificationDataModule.from_dataset_dict(dataset_dict, tokenizer=tokenizer)

    # load model using data properties
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model.name_or_path,
        id2label=datamodule.id2label,
        label2id=datamodule.label2id,
        num_labels=len(datamodule.labels),
    )

    # define estimator
    estimator = EstimatorForSequenceClassification(model=model, **cfg.trainer)

    # sanity check

    # fit
    train_outputs = estimator.fit(
        train_loader=datamodule.train_dataloader(),
        validation_loader=datamodule.val_dataloader(),
        **cfg.fit,
    )

    # test
    test_outputs = estimator.test(datamodule.test_dataloader(), **cfg.test)

    # save experiment output
    with open("./train_outputs", "wb") as fl:
        pickle.dump(train_outputs, fl)

    with open("./test_outputs", "wb") as fl:
        pickle.dump(test_outputs, fl)


if __name__ == "__main__":
    main()
