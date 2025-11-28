import logging

import pandas as pd
import streamlit as st

from src.config import DATA_PATH  # uses the path you defined in config.py

# -------------------------------------------------------------------
# Logging configuration
# -------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)


# -------------------------------------------------------------------
# Data helpers
# -------------------------------------------------------------------
@st.cache_data
def load_data() -> pd.DataFrame | None:
    """Load the Mumbai housing dataset defined in DATA_PATH."""
    try:
        df = pd.read_csv(DATA_PATH)

        # Drop any unnamed index columns
        unnamed = [c for c in df.columns if c.lower().startswith("unnamed")]
        if unnamed:
            df = df.drop(columns=unnamed)

        return df
    except FileNotFoundError:
        logger.warning("Data file not found at %s", DATA_PATH)
        return None
    except Exception as e:
        logger.error("Failed to load data: %s", e)
        return None


@st.cache_data
def get_locations() -> list[str]:
    """
    Return sorted unique locations from the dataset.

    This is what drives the Location selectbox.
    """
    df = load_data()
    if df is None or "Location" not in df.columns:
        logger.warning("No Location column found; falling back to ['Mumbai'].")
        return ["Mumbai"]

    locs = (
        df["Location"]
        .dropna()
        .astype(str)
        .str.strip()
        .replace("", pd.NA)
        .dropna()
        .unique()
        .tolist()
    )
    locs = sorted(set(locs))
    return locs if locs else ["Mumbai"]


def predict_dummy_price(row: dict) -> float:
    """
    Dummy price estimator so the UI works even without a real model.

    Replace this with your real model.predict() when you’re ready.
    """
    base = 15_00_0000  # base price
    base += row["Area"] * 2000
    base += row["No. of Bedrooms"] * 10_00_000
    base += (row["Gymnasium"] + row["Swimming Pool"]) * 5_00_000
    return float(base)


# -------------------------------------------------------------------
# Streamlit UI
# -------------------------------------------------------------------
def main():
    st.set_page_config(
        page_title="Mumbai Real Estate Price Prediction",
        layout="wide",
    )

    st.title("Mumbai Real Estate Price Prediction")
    st.write(
        "Enter property details below to get an estimated price based on a "
        "machine learning model trained on Mumbai housing data."
    )

    df = load_data()
    locations = get_locations()

    # ---------------- FORM ----------------
    st.header("Property details")
    col1, col2 = st.columns(2)

    with col1:
        area = st.number_input(
            "Area (sq ft)",
            min_value=100.0,
            max_value=10000.0,
            value=1500.0,
            step=50.0,
        )
        bedrooms = st.number_input(
            "No. of Bedrooms",
            min_value=1,
            max_value=10,
            value=4,
            step=1,
        )
        location = st.selectbox(
            "Location",
            options=locations,
            index=0,
            help="Start typing to search across all locations in the dataset.",
        )

    with col2:
        new_resale = st.selectbox("New or Resale", ["Resale", "New"])
        gym = st.checkbox("Gymnasium", value=True)
        lift = st.checkbox("Lift Available", value=True)
        parking = st.checkbox("Car Parking", value=True)
        maint = st.checkbox("Maintenance Staff", value=True)
        sec24 = st.checkbox("24x7 Security", value=True)
        play = st.checkbox("Children's Play Area", value=True)
        club = st.checkbox("Clubhouse", value=True)
        intercom = st.checkbox("Intercom", value=True)
        garden = st.checkbox("Landscaped Gardens")
        indoor = st.checkbox("Indoor Games")
        gas = st.checkbox("Gas Connection", value=True)
        track = st.checkbox("Jogging Track", value=True)
        pool = st.checkbox("Swimming Pool", value=True)

    # Single row for model / dummy input
    row = {
        "Area": area,
        "No. of Bedrooms": bedrooms,
        "New/Resale": 1 if new_resale == "New" else 0,
        "Gymnasium": int(gym),
        "Lift Available": int(lift),
        "Car Parking": int(parking),
        "Maintenance Staff": int(maint),
        "24x7 Security": int(sec24),
        "Children's Play Area": int(play),
        "Clubhouse": int(club),
        "Intercom": int(intercom),
        "Landscaped Gardens": int(garden),
        "Indoor Games": int(indoor),
        "Gas Connection": int(gas),
        "Jogging Track": int(track),
        "Swimming Pool": int(pool),
        "Location": location,
    }

    st.write("")  # spacer

    if st.button("Predict Price"):
        # === replace this with your real model when ready ===
        price = predict_dummy_price(row)

        st.subheader(f"Estimated Price: ₹{price:,.0f}")

        if df is not None and "Price" in df.columns:
            avg_price = df["Price"].mean()
            st.caption(f"Average price in dataset: ₹{avg_price:,.0f}")

        st.caption("Note: This tool is for educational use only, not financial advice.")


if __name__ == "__main__":
    main()



