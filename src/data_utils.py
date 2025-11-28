import logging
from typing import Tuple

import pandas as pd

from .config import DATA_PATH, TARGET_COL

logger = logging.getLogger(__name__)


def load_raw_data() -> pd.DataFrame:
    """
    Load the raw Mumbai housing dataset from disk.
    """
    logger.info("Loading data from %s", DATA_PATH)
    df = pd.read_csv(DATA_PATH)

    # Drop unnamed index columns if present
    unnamed = [c for c in df.columns if c.lower().startswith("unnamed")]
    if unnamed:
        logger.info("Dropping unnamed columns: %s", unnamed)
        df = df.drop(columns=unnamed)

    return df


def train_test_split(
    df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Simple train/holdout split.
    """
    from sklearn.model_selection import train_test_split as sk_split

    logger.info("Splitting data into train and test (test_size=%s)", test_size)
    train_df, test_df = sk_split(df, test_size=test_size, random_state=random_state)
    return train_df, test_df


def split_features_target(df: pd.DataFrame):
    """
    Separate features and target.
    """
    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found in dataframe.")

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    return X, y
