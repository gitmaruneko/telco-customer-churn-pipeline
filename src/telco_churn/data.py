import shutil
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.request import urlopen

import pandas as pd

DATASET_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
    "master/data/Telco-Customer-Churn.csv"
)


def load_raw_dataset(path: str | Path) -> pd.DataFrame:
    """Load the raw CSV without hiding data-quality issues from exploration."""
    return pd.read_csv(path, keep_default_na=False)


def download_raw_dataset(path: str | Path) -> None:
    """Download the IBM sample dataset to the given path."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None

    try:
        with NamedTemporaryFile(
            mode="wb",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            with urlopen(DATASET_URL, timeout=30) as response:
                shutil.copyfileobj(response, temporary_file)

        if temporary_path.stat().st_size == 0:
            raise OSError("Downloaded dataset is empty")

        temporary_path.replace(destination)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
