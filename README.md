# Mumbai Real Estate Price Prediction

This project predicts the price of residential properties in Mumbai using a machine learning model trained on public housing data.

The goal of the project is to:
- Clean and explore a raw real-estate dataset,
- Engineer useful features,
- Train and evaluate a regression model,
- Expose the model through a simple Streamlit web app that non-technical users can try.

---

## 1. Project Structure

```text
.
├── data/
│   └── Mumbai1.csv               # raw dataset
├── models/
│   └── house_price_model.pkl     # trained regression pipeline
├── notebooks/
│   └── real_estate_model.ipynb   # exploratory analysis & experimentation
├── app.py                        # Streamlit app
├── model_training.py             # training script
├── requirements.txt
└── README.md
