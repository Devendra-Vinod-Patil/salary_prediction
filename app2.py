import streamlit as st
import pickle
import joblib
import pandas as pd

# ------------------------------
# Load model and features
# ------------------------------
with open("final_model.pkl", "rb") as f:
    model = pickle.load(f)

feature_names = joblib.load("feature_names.pkl")

st.set_page_config(page_title="Salary Prediction", page_icon="💼", layout="centered")
st.title("💼 Salary Prediction App")
st.write("Predict the correct salary for Indian non-metro jobs using a trained ML model.")

# ------------------------------
# Input form
# ------------------------------
st.header("Enter Job Features")

user_input = {}

for feature in feature_names:
    # Heuristic: choose widget type
    if any(keyword in feature.lower() for keyword in ["job", "role", "edu", "location", "city"]):
        user_input[feature] = st.text_input(f"{feature}", "")
    else:
        user_input[feature] = st.number_input(f"{feature}", value=0.0)

# Convert input dict to dataframe with same column order as training
input_df = pd.DataFrame([[user_input[feat] for feat in feature_names]], columns=feature_names)

# ------------------------------
# Prediction
# ------------------------------
if st.button("Predict Salary"):
    try:
        salary = model.predict(input_df)
        st.success(f"💰 Predicted Salary: ₹ {salary[0]:,.2f}")
    except Exception as e:
        st.error(f"⚠️ Error during prediction: {e}")
