"""
Entry point for training the Mumbai house price model.

This script simply calls the core training function in src.train
so that external tools (and Streamlit Cloud) can still run
`python model_training.py` as before.
"""

import logging

from src.train import train_model


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )
    train_model()
