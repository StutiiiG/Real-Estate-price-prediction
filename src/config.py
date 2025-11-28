from pathlib import Path

# Base project directory (root of the repo)
BASE_DIR = Path(__file__).resolve().parents[1]

# Data & model paths
DATA_PATH = BASE_DIR / "notebooks" / "Mumbai1.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "house_price_model.pkl"

# Target column in the dataset
TARGET_COL = "Price"
