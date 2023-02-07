from pathlib import Path

import srsly
from datasets import load_from_disk

from src.data.active_datamodule import ActiveClassificationDataModule
from src.transformers import UncertaintyBasedStrategyForSequenceClassification
from transformers import AutoModelForSequenceClassification, AutoTokenizer, logging

logging.set_verbosity_error()

if __name__ == "__main__":

    data_path = Path("data/prepared/ag_news")
    dataset_dict = load_from_disk(data_path)
    metadata = srsly.read_yaml(data_path / "metadata.yaml")
    tokenizer = AutoTokenizer.from_pretrained(metadata["name_or_path"])
    active_datamodule = ActiveClassificationDataModule.from_dataset_dict(
        dataset_dict,
        tokenizer=tokenizer,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        metadata["name_or_path"],
        num_labels=len(active_datamodule.labels),
        id2label=active_datamodule.id2label,
        label2id=active_datamodule.label2id,
    )

    active_estimator = UncertaintyBasedStrategyForSequenceClassification(model, score_fn="margin_confidence")

    active_out = active_estimator.active_fit(
        active_datamodule=active_datamodule,
        num_rounds=3,
        query_size=50,
        val_perc=0.3,
        fit_kwargs={"num_epochs": 3, "limit_train_batches": 3, "limit_validation_batches": 3},
        test_kwargs={"limit_batches": 3},
        pool_kwargs={"limit_batches": 3},
        save_dir="notebooks/results",
    )
