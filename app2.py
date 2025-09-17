import streamlit as st
import pandas as pd
import joblib
import os

# =========================
# Load trained artifacts
# =========================
base_path = os.path.dirname(__file__)

model = joblib.load(os.path.join(base_path, "final_model.pkl"))
rfe = joblib.load(os.path.join(base_path, "rfe.pkl"))
feature_names = joblib.load(os.path.join(base_path, "feature_names.pkl"))
encoders = joblib.load(os.path.join(base_path, "encoders.pkl"))  # dict of LabelEncoders/OneHot for categorical features

# =========================
# Streamlit App
# =========================
st.set_page_config(page_title="💰 Employee Salary Prediction", layout="wide")
st.title("💰 Employee Salary Prediction")
st.write("Enter employee details to predict the salary based on experience, skills, location, company, and role.")

# -------------------------
# Input Fields
# -------------------------
experience = st.slider("Experience (years)", min_value=0, max_value=40, value=5, step=1)
state = st.selectbox("State", options=encoders['state'].classes_.tolist())
company = st.selectbox("Company", options=encoders['company'].classes_.tolist())
role = st.selectbox("Role / Job Title", options=encoders['role'].classes_.tolist())
skills = st.multiselect("Skills", options=encoders['skills'].classes_.tolist())

# -------------------------
# Predict Button
# -------------------------
if st.button("Predict Salary"):
    # Build input dict
    input_dict = {}

    # Numeric features
    input_dict['experience'] = [experience]

    # Encode categorical features
    input_dict['state'] = [encoders['state'].transform([state])[0]]
    input_dict['company'] = [encoders['company'].transform([company])[0]]
    input_dict['role'] = [encoders['role'].transform([role])[0]]

    # Skills multi-hot encoding
    all_skills = encoders['skills'].classes_
    skill_vector = [1 if s in skills else 0 for s in all_skills]
    for i, s in enumerate(all_skills):
        input_dict[f"skill_{s}"] = [skill_vector[i]]

    # Convert to DataFrame
    input_df = pd.DataFrame(input_dict)

    # Select only features used in RFE
    input_df_rfe = input_df[feature_names]
    transformed = rfe.transform(input_df_rfe)

    # Predict salary
    prediction = model.predict(transformed)
    st.success(f"💵 Predicted Salary: {prediction[0]:,.2f}")
