import streamlit as st
import pandas as pd
import joblib
import numpy as np

# ---- Load model and preprocessing objects ----
model_data = joblib.load("joblib.jb")
model = model_data["model"]
scaler = model_data["scaler"]
encoders = model_data["encoders"]       # dict: {col: LabelEncoder}
feature_names = model_data["features"]  # list of columns in exact training order

# ---- Page ----
st.title("📊 Customer Churn Prediction")
st.header("Enter Customer Details")

# ---- Build input dict using the saved features and encoders ----
input_data = {}

# Categorical features: use selectbox to avoid unseen-label errors
for col, le in encoders.items():
    input_data[col] = st.selectbox(f"{col}", list(le.classes_))

# Numerical features: those in feature_names that are not in encoders
for col in feature_names:
    if col not in encoders.keys():
        # Use step and format if you want; keep simple
        input_data[col] = st.number_input(f"{col}", value=0.0, format="%.4f")

# Convert to DataFrame (single-row)
input_df = pd.DataFrame([input_data])

# ---- Predict button (do all processing and prediction inside this block) ----
if st.button("🔮 Predict Churn"):
    # 1) Encode categorical columns (LabelEncoder)
    for col, le in encoders.items():
        # we used selectbox so value is guaranteed to be in le.classes_
        input_df[col] = le.transform([input_df[col][0]])

    # 2) Reorder columns exactly as during training
    try:
        input_df = input_df[feature_names]
    except KeyError as e:
        st.error(f"Feature mismatch: {e}")
        st.stop()

    # 3) Ensure numeric dtypes for scaler
    #    Convert every column to float (scaler expects numeric arrays)
    #    (Label encoded columns become integers but safe to convert to float)
    input_df = input_df.astype(float)

    # 4) Scale with the saved scaler
    try:
        scaled_input = scaler.transform(input_df)
    except Exception as e:
        st.error(f"Error while scaling input: {e}")
        st.stop()

    # 5) Predict - try to show probabilities if model supports it
    try:
        proba = model.predict_proba(scaled_input)[0]  # [prob_not_churn, prob_churn]
    except Exception:
        proba = None

    prediction = model.predict(scaled_input)[0]

    # 6) Show results (probabilities if available)
    st.subheader("Prediction Result")
    if proba is not None:
        # If binary but ordering could be [class0, class1], we will also show which is churn class
        # assume during training Churn was encoded as 1 and Not Churn as 0 (common). If different,
        # user should adapt labels.
        not_churn_prob = proba[0]
        churn_prob = proba[1] if len(proba) > 1 else 1 - not_churn_prob
        st.write(f"🔵 Probability of Not Churning: **{not_churn_prob*100:.2f}%**")
        st.write(f"🔴 Probability of Churning: **{churn_prob*100:.2f}%**")
    else:
        st.info("Model does not support probability output; showing class prediction only.")

    
