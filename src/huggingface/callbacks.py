from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
import srsly
from lightning.fabric.wrappers import _FabricModule

from src.callbacks import Callback
from src.enums import OutputKeys, RunningStage, SpecialKeys
from src.estimator import Estimator
from src.types import EPOCH_OUTPUT, METRIC
from src.utilities import move_to_cpu


class SaveOutputs(Callback):
    def __init__(self, dirpath: Union[str, Path], instance_level: bool, batch_level: bool, epoch_level: bool) -> None:
        super().__init__()
        self.dirpath = Path(dirpath)
        self.instance_level = instance_level
        self.batch_level = batch_level
        self.epoch_level = epoch_level

    def on_epoch_end(
        self,
        estimator: Estimator,
        model: _FabricModule,
        output: EPOCH_OUTPUT,
        metrics: METRIC,
        stage: RunningStage,
        **kwargs,
    ) -> None:
        # output directory setup
        suffix = f"_epoch_{estimator.counter.num_epochs}" if stage != RunningStage.TEST else ""
        path = self.dirpath / f"{stage}"
        path.mkdir(exist_ok=True, parents=True)

        # list of dicts to dict of lists
        data = {k: [dic[k] for dic in output] for k in output[0]}

        # instance-level output
        if self.instance_level:
            logits = np.concatenate(data.pop(OutputKeys.LOGITS))
            idx = np.concatenate(data.pop(SpecialKeys.ID))
            instance_level_output = pd.DataFrame(logits, columns=[f"logit_{i}" for i in range(logits.shape[1])]).assign(
                **{SpecialKeys.ID: idx}
            )
            instance_level_output.to_parquet(path / f"instance_level{suffix}.parquet", index=False)

        # batch-level output
        if self.batch_level:
            batch_metrics = data.pop(OutputKeys.METRICS)
            batch_metrics = {k: [dic[k] for dic in batch_metrics] for k in batch_metrics[0]}
            batch_level_output = pd.DataFrame(batch_metrics).assign(loss=data[OutputKeys.LOSS])
            batch_level_output.to_parquet(path / f"batch_level{suffix}.parquet", index=False)

        # epoch-level output
        if self.epoch_level:
            total_metrics = move_to_cpu(metrics.compute())
            epoch_level_output = {
                OutputKeys.METRICS: total_metrics,
                OutputKeys.LOSS: round(np.mean(data[OutputKeys.LOSS]), 6),
            }
            srsly.write_json(path / f"epoch_level{suffix}.json", epoch_level_output)

    def on_train_epoch_end(
        self, estimator: Estimator, model: _FabricModule, output: EPOCH_OUTPUT, metrics: METRIC, **kwargs
    ) -> None:
        return self.on_epoch_end(estimator, model, output, metrics, RunningStage.TRAIN, **kwargs)

    def on_validation_epoch_end(
        self, estimator: Estimator, model: _FabricModule, output: EPOCH_OUTPUT, metrics: METRIC, **kwargs
    ) -> None:
        return self.on_epoch_end(estimator, model, output, metrics, RunningStage.VALIDATION, **kwargs)

    def on_test_epoch_end(
        self, estimator: Estimator, model: _FabricModule, output: EPOCH_OUTPUT, metrics: METRIC, **kwargs
    ) -> None:
        return self.on_epoch_end(estimator, model, output, metrics, RunningStage.TEST, **kwargs)
