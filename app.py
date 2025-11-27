import streamlit as st
import pandas as pd
import joblib
import os

MODEL_PATH = "models/house_price_model.pkl"
DATA_PATH = "Mumbai1.csv"

@st.cache_data
def load_data():
    if os.path.exists(DATA_PATH):
        return pd.read_csv(DATA_PATH)
    return None

@st.cache_resource
def load_model():
    obj = joblib.load(MODEL_PATH)
    return obj["pipeline"], obj["features"]

def main():
    st.title("Mumbai Real Estate Price Prediction")
    st.write(
        "Enter property details below to get an estimated price based on a machine "
        "learning model trained on Mumbai housing data."
    )

    df = load_data()
    model, feature_cols = load_model()

    # --- UI inputs (replace with your real features) ---
    # These names MUST match feature_cols you used in model_training.py
    col1, col2, col3 = st.columns(3)

    with col1:
        area = st.number_input("Area (sq ft)", min_value=100.0, max_value=5000.0, value=800.0, step=50.0)
    with col2:
        bedrooms = st.number_input("Bedrooms", min_value=1, max_value=6, value=2, step=1)
    with col3:
        bathrooms = st.number_input("Bathrooms", min_value=1, max_value=6, value=1, step=1)

    # Build input row in the same order as feature_cols
    input_dict = {
        "area": area,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
    }
    # Filter to the columns model expects
    X_input = pd.DataFrame([input_dict])[feature_cols]

    if st.button("Predict Price"):
        price_pred = model.predict(X_input)[0]
        st.subheader(f"Estimated Price: ₹{price_pred:,.0f}")

        if df is not None and "price" in df.columns:
            avg_price = df["price"].mean()
            st.write(f"Average price in dataset: ₹{avg_price:,.0f}")

        st.caption("Note: This is an educational demo, not financial advice.")

if __name__ == "__main__":
    main()
