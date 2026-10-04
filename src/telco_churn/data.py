from pathlib import Path

import pandas as pd

def load_raw_dataset(path: str | Path) -> pd.DataFrame:
    """Load the raw CSV without hiding data-quality issues from exploration."""
    return pd.read_csv(path, keep_default_na=False)
