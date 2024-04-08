import argparse
from pathlib import Path

from tbparse import SummaryReader
from tqdm.auto import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str)
    args = parser.parse_args()

    path = Path(args.dir)
    for d in path.iterdir():
        if not d.is_dir():
            continue

        for p in tqdm(list((d).iterdir()), desc=d.name):
            if (p / "tensorboard_logs.parquet").exists() or not (p / "tb_logs").exists():
                continue

            tb_logs_path = p / "tb_logs"
            logs = SummaryReader(str(tb_logs_path))
            logs.scalars.to_parquet(p / "tensorboard_logs.parquet")
