import logging
import pickle
from pathlib import Path

import hydra
from datasets import load_from_disk
from hydra.utils import get_original_cwd
from lightning.fabric import seed_everything
from omegaconf import DictConfig, OmegaConf

from src.data.active_datamodule import ActiveClassificationDataModule
from src.logging import set_ignore_warnings
from src.query_strategies.classification import UncertantyBasedStrategy
from transformers import AutoModelForSequenceClassification, AutoTokenizer

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
    datamodule = ActiveClassificationDataModule.from_dataset_dict(dataset_dict, tokenizer=tokenizer)

    # load model using data properties
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model.name_or_path,
        id2label=datamodule.id2label,
        label2id=datamodule.label2id,
        num_labels=len(datamodule.labels),
    )

    # define estimator
    active_estimator = UncertantyBasedStrategy(score_fn="margin_confidence", model=model, **cfg.trainer)

    # active learning loop
    active_estimator.active_fit(
        active_datamodule=datamodule,
        num_rounds=5,
        query_size=50,
        val_perc=None,
        fit_kwargs=cfg.fit,
        test_kwargs=cfg.test,
        pool_kwargs=cfg.test,
    )


if __name__ == "__main__":
    main()
