from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
import srsly
from lightning.fabric.wrappers import _FabricModule

from src.callbacks import Callback
from src.enums import OutputKeys, RunningStage, SpecialKeys
from src.types import EPOCH_OUTPUT, METRIC
from src.utilities import move_to_cpu


class SaveOutputs(Callback):
    def __init__(self, dirpath: Union[str, Path]) -> None:
        super().__init__()
        self.dirpath = Path(dirpath)

    def on_epoch_end(
        self, model: _FabricModule, output: EPOCH_OUTPUT, metrics: METRIC, stage: RunningStage, **kwargs
    ) -> None:
        epoch_idx = kwargs.get("epoch_idx", None)
        suffix = f"_epoch_{epoch_idx}" if epoch_idx is not None else ""

        # list of dicts to dict of lists
        data = {k: [dic[k] for dic in output] for k in output[0]}

        # instance-level output
        logits = np.concatenate(data.pop(OutputKeys.LOGITS))
        idx = np.concatenate(data.pop(SpecialKeys.ID))
        instance_level_output = pd.DataFrame(logits, columns=[f"logit_{i}" for i in range(logits.shape[1])]).assign(
            **{SpecialKeys.ID: idx}
        )

        # batch-level output
        loss = data.pop(OutputKeys.LOSS)
        batch_metrics = data.pop(OutputKeys.METRICS)
        batch_metrics = {k: [dic[k] for dic in batch_metrics] for k in batch_metrics[0]}
        batch_level_output = pd.DataFrame(batch_metrics).assign(loss=loss)

        # epoch-level output
        total_metrics = move_to_cpu(metrics.compute())
        epoch_level_output = {
            "metrics": total_metrics,
            "loss": round(np.mean(loss), 6),
        }

        # save
        path = self.dirpath / f"{stage}"
        path.mkdir(exist_ok=True, parents=True)

        instance_level_output.to_parquet(path / f"instance_level{suffix}.parquet", index=False)
        batch_level_output.to_parquet(path / f"batch_level{suffix}.parquet", index=False)
        srsly.write_json(path / f"epoch_level{suffix}.json", epoch_level_output)

    def on_train_epoch_end(self, model: _FabricModule, output: EPOCH_OUTPUT, metrics: METRIC, **kwargs) -> None:
        return self.on_epoch_end(model, output, metrics, RunningStage.TRAIN, **kwargs)

    def on_validation_epoch_end(self, model: _FabricModule, output: EPOCH_OUTPUT, metrics: METRIC, **kwargs) -> None:
        return self.on_epoch_end(model, output, metrics, RunningStage.VALIDATION, **kwargs)

    def on_test_epoch_end(self, model: _FabricModule, output: EPOCH_OUTPUT, metrics: METRIC, **kwargs) -> None:
        return self.on_epoch_end(model, output, metrics, RunningStage.TEST, **kwargs)
